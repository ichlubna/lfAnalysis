import lfreader as lfr
import evaluator as eva
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from progress.bar import ChargingBar
import numpy
import cv2
import sys
import traceback

class KernelParams:
    name = ""
    module = None
    parameter = 0
    wgDivide = (1,1)

    def __init__(self, name, module, parameter, wgDivide):
        self.name = name
        self.module = module
        self.parameter = parameter
        self.wgDivide = wgDivide

class KernelTester:
    imageCount = 64
    numberOfMeasurements = 10
    width = 0
    height = 0
    cols = 0
    rows = 0
    depth = 4
    size = 0
    totalSize = 0
    lfReader = None
    imagesGPU = None
    resultGPU = None
    kernels = []

    def loadInput(self):
        lfReader = lfr.LfReader()
        lfReader.setPixFmt("RGBA")
        path = sys.argv[1]
        lfReader.loadDir(path)
        colsRows = lfReader.getColsRows()
        self.cols = numpy.int32(colsRows[0])
        self.rows = numpy.int32(colsRows[1])
        self.imageCount = self.cols*self.rows
        width, height = lfReader.getResolution()
        self.width = numpy.int32(width)
        self.height = numpy.int32(height)
        depth = numpy.int32(4)
        self.size = int(width*height*depth)
        self.totalSize = numpy.uint(self.size*self.imageCount)
        self.lfReader = lfReader

    def allocGPUResources(self):
        bar = ChargingBar("Allocating and uploading textures", max=self.cols*self.rows+1)
        images = bytes()
        for y in range(0,self.rows):
            for x in range(0, self.cols):
                img = self.lfReader.openImage(y,x)
                imgBytes = img.tobytes()
                images += imgBytes
                bar.next()
        bar.finish()

        self.imagesGPU = cuda.mem_alloc(int(self.totalSize))
        cuda.memcpy_htod(self.imagesGPU, images)
        self.resultGPU = cuda.mem_alloc(self.size)
        bar.next()
        bar.finish()

    def compileKernels(self):
        kernelConstants = ["-DIMG_WIDTH="+str(self.width), "-DIMG_HEIGHT="+str(self.height), "-DGRID_COLS="+str(self.cols), "-DGRID_ROWS="+str(self.rows)]

        kernelSourceGeneral = """
        typedef struct {float r,g,b;} Pixel;
        __device__ uint2 getImgCoords()
        {
            uint2 coords;
            coords.x = (threadIdx.x + blockIdx.x * blockDim.x);
            coords.y = (threadIdx.y + blockIdx.y * blockDim.y);
            return coords;
        }

        __device__ unsigned int getLinearID(uint2 coords, int width)
        {
            return width*coords.y + coords.x;
        }

        __device__ float squaredDistance(float2 a, float2 b)
        {
            return powf(a.x-b.x,2) + powf(a.y-b.y,2);
        }

        __device__ uint2 focusCoords(uint2 coords, int focus, uint2 position, float2 center)
        {
            float aspect = float(IMG_WIDTH)/IMG_HEIGHT;
            float2 offset{center.x-position.x, center.y-position.y};
            return {focus*offset.x, round(aspect*offset.y*focus)};
        }

        class Images
        {
            public:
            uchar4 *inData[GRID_COLS*GRID_ROWS];
            uchar4 *outData;
            int width = 0;
            int height = 0;
            __device__ Images(int w, int h) : width{w}, height{h} {};

            __device__ uchar4 getPixel(int imageID, uint2 coords)
            {
                uint2 clamped{min(coords.x, IMG_WIDTH), min(coords.y, IMG_HEIGHT)};
                unsigned int linearCoord = getLinearID(clamped, IMG_WIDTH);
                return inData[imageID][linearCoord];
            }

            __device__ void setPixel(uint2 coords, uchar4 pixel)
            {
                unsigned int linearCoord = getLinearID(coords, IMG_WIDTH);
                outData[linearCoord] = pixel;
            }
        };

        """

        kernelSourceMain = """
        __global__ void process(unsigned char inputImages[GRID_COLS*GRID_ROWS][IMG_WIDTH*IMG_HEIGHT], unsigned char *result, int parameter)
          {
            Images images(IMG_WIDTH, IMG_HEIGHT);
            for(int i=0; i<GRID_COLS*GRID_ROWS; i++)
                images.inData[i] = reinterpret_cast<uchar4*>(inputImages[i]);
            images.outData = reinterpret_cast<uchar4*>(result);

            uint2 coords = getImgCoords();
            if(coords.x >= IMG_WIDTH || coords.y >= IMG_HEIGHT)
                return;

            interpolateImages(images, result, coords, parameter);
          }
        """

        kernelSourcePerPixel = """
        __device__ void interpolateImages(Images images, unsigned char *result, uint2 coords, int focus)
        {
            float sum[]{0,0,0,0};
            float2 gridCenter{GRID_COLS/2.f, GRID_ROWS/2.f};
            float maxDistance = squaredDistance({0,0}, gridCenter);
            float weightSum{0};
            for(unsigned int y = 0; y<GRID_ROWS; y++)
                for(unsigned int x = 0; x<GRID_COLS; x++)
                {
                    uint2 focusedCoords = focusCoords(coords, 50, {x,y}, gridCenter);
                    float weight = 1.0f;// 1.f-maxDistance/squaredDistance({float(x),float(y)}, gridCenter);
                    weightSum += weight;
                    int gridID = getLinearID({x,y}, GRID_COLS);
                    uchar4 pixel = images.getPixel(gridID, focusedCoords);
                    float fPixel[]{float(pixel.x), float(pixel.y), float(pixel.z), float(pixel.w)};
                    for(int j=0; j<4; j++)
                        sum[j] = __fmaf_rn(fPixel[j], weight, sum[j]);
                }
            for(int j=0; j<4; j++)
                sum[j] /= weightSum;
            uchar4 chSum{sum[0], sum[1], sum[2], sum[3]};
            images.setPixel(coords, chSum);
        }
        """


        perPixelKernel = SourceModule(kernelSourceGeneral+kernelSourcePerPixel+kernelSourceMain, options=kernelConstants)

        self.kernels = [ KernelParams("Per pixel", perPixelKernel, numpy.int32(1), (1, 1))]

    def runKernels(self):
        result = numpy.zeros((self.height, self.width, self.depth), numpy.uint8)
        for kernel in self.kernels:
            print(kernel.name)
            for i in range(0,self.numberOfMeasurements):
                start=cuda.Event()
                end=cuda.Event()
                start.record()
                func = kernel.module.get_function("process")
                func(self.imagesGPU, self.resultGPU, kernel.parameter, block=(16, 16, 1), grid=(int(self.width/(16/kernel.wgDivide[0])), int(self.height/(16/kernel.wgDivide[1]))), shared=0)
                end.record()
                end.synchronize()
                print("Time: "+str(start.time_till(end))+" ms")

                if i == self.numberOfMeasurements-1:
                    cuda.memcpy_dtoh(result, self.resultGPU)
                    result = result.astype(numpy.uint8)
                    resultImage = Image.frombytes("RGBA", (self.width, self.height), result)
                    resultImage.save("./distorted/test.png")
                    evaluator = eva.Evaluator()
                    print(evaluator.metrics("./original", "./distorted"))
                    print("")

try:
    kt = KernelTester()
    kt.loadInput()
    kt.compileKernels()
    kt.allocGPUResources()
    kt.runKernels()
except Exception as e:
    print(e)
    print(traceback.format_exc())
