import lfreader as lfr
import evaluator as eva
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
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
    imageCount = 4
    numberOfMeasurements = 10
    width = 0
    height = 0
    depth = 4
    size = 0
    lfReader = None
    imagesGPU = None
    resultGPU = None
    leftTopX = 0
    leftTopY = 0
    kernels = []

    def loadInput(self):
        lfReader = lfr.LfReader()
        lfReader.setPixFmt("RGBA")
        path = sys.argv[1]
        lfReader.loadDir(path)
        colsRows = lfReader.getColsRows()
        self.leftTopX = int(colsRows[0]/2)-1
        self.leftTopY = int(colsRows[1]/2)-1
        width, height = lfReader.getResolution()
        self.width = numpy.int32(width)
        self.height = numpy.int32(height)
        depth = numpy.int32(4)
        self.size = int(width*height*depth)
        self.lfReader = lfReader

    def allocGPUResources(self):
        images = bytes()
        for i in range(0,self.imageCount):
            img = self.lfReader.openImage(self.leftTopY+i%2, int(self.leftTopX+i/2))
            imgBytes = img.tobytes()
            images = images + imgBytes

        self.imagesGPU = cuda.mem_alloc(self.imageCount*self.size)
        cuda.memcpy_htod(self.imagesGPU, images)
        self.resultGPU = cuda.mem_alloc(int(self.size))

    def compileKernels(self):
        kernelConstants = ["-DIMG_SIZE="+str(self.size), "-DIMG_COUNT="+str(self.imageCount)]

        kernelSourceGeneral = """
        typedef struct {float r,g,b;} Pixel;
        __device__ uint2 getImgCoords()
        {
            uint2 coords;
            coords.x = (threadIdx.x + blockIdx.x * blockDim.x);
            coords.y = (threadIdx.y + blockIdx.y * blockDim.y);
            return coords;
        }

        class Images
        {
            public:
            uchar4 *inData[IMG_COUNT];
            uchar4 *outData;
            int width = 0;
            int height = 0;
            __device__ Images(int w, int h) : width{w}, height{h} {};

            __device__ uchar4 getPixel(int id, int linearCoord)
            {
                return inData[id][linearCoord];
            }

            __device__ void setPixel(int linearCoord, uchar4 pixel)
            {
                outData[linearCoord] = pixel;
            }

            __device__ unsigned int getLinearID(uint2 coords)
            {
                return width*coords.y + coords.x;
            }
        };

        """

        kernelSourceMain = """
        __global__ void process( int width, int height, unsigned char inputImages[IMG_COUNT][IMG_SIZE], unsigned char *result, int parameter)
          {
            Images images(width, height);
            for(int i=0; i<IMG_COUNT; i++)
                images.inData[i] = reinterpret_cast<uchar4*>(inputImages[i]);
            images.outData = reinterpret_cast<uchar4*>(result);

            uint2 coords = getImgCoords();
            if(coords.x >= width || coords.y >= height)
                return;
            interpolateImages(images, result, coords, parameter);

          }
        """

        kernelSourcePerPixel = """
        __device__ void interpolateImages(Images images, unsigned char *result, uint2 coords, int pixelCount)
        {
            int id  = images.getLinearID(coords);
            float weights[]{0.2f, 0.5f, 0.1f, 0.2f};
            for(int p = 0; p<pixelCount; p++)
            {
                int newId = id+p*+p*(IMG_SIZE/pixelCount);
                //int newId = pixelCount*id+p;
                float sum[]{0,0,0,0};
                for(int i = 0; i<IMG_COUNT; i++)
                {
                    uchar4 pixel = images.getPixel(i, newId);
                    float fPixel[]{float(pixel.x), float(pixel.y), float(pixel.z), float(pixel.w)};
                    for(int j=0; j<4; j++)
                        //sum[j] += fPixel[j] * weights[i];
                        sum[j] = __fmaf_rn(fPixel[j], weights[i], sum[j]);
                }
                uchar4 chSum{sum[0], sum[1], sum[2], sum[3]};
                images.setPixel(newId, chSum);
            }
        }
        """

        kernelSourcePerBlock = """

        __device__ void loadBlock(unsigned char (*block)[4][4], unsigned char img[IMG_SIZE], int id)
        {
            int *imgInt = reinterpret_cast<int*>(img);
            const int pixelCount = 4;
            for(int i=0; i<pixelCount; i++)
            {
                int *blockInt = reinterpret_cast<int*>((*block)[i]);
                blockInt[i] = imgInt[id+i];
            }
        }

        __device__ void storeBlock(float (*block)[4][4], unsigned char *result, int id)
        {
            int *resultInt = reinterpret_cast<int*>(result);
            const int pixelCount = 4;
            for(int i=0; i<pixelCount; i++)
            {
                unsigned char pixel[4];
                for(int j=0; j<4; j++)
                    pixel[j] = int((*block)[i][j]);
                int *pixelInt = reinterpret_cast<int*>(&pixel);
                resultInt[id+i] = *pixelInt;
            }
        }

        __device__ void interpolateImages(unsigned char images[IMG_COUNT][IMG_SIZE], int id, unsigned char *result)
        {
            float weights[]{0.2f, 0.5f, 0.1f, 0.2f};
            const int pixelCount = 4;
            const int depth = 4;
            for(int i = 0; i<IMG_COUNT; i++)
            {
                unsigned char block[4][4];
                loadBlock(&block, images[i], id);
                float sum[4][4]{0,0,0,0};
                for(int p = 0; p<pixelCount; p++)
                {
                    float pixel[]{float(block[p][0]), float(block[p][1]), float(block[p][2]), float(block[p][3])};
                    for(int j=0; j<depth; j++)
                        sum[p][j] = __fmaf_rn(pixel[j], weights[i], sum[p][j]);
                }
                storeBlock(&sum, result, id);
            }
        }
        """

        perPixelKernel = SourceModule(kernelSourceGeneral+kernelSourcePerPixel+kernelSourceMain, options=kernelConstants)
        #perBlockKernel = SourceModule(kernelSourceGeneral+kernelSourcePerBlock+kernelSourceMain, options=kernelConstants)

        self.kernels = [ KernelParams("Per pixel", perPixelKernel, numpy.int32(1), (1, 1))]
                    #KernelParams("Per block", perBlockKernel, numpy.int32(4), (4, 1))]

    def runKernels(self):
        result = numpy.zeros((self.height, self.width, self.depth), numpy.uint8)
        for kernel in self.kernels:
            print(kernel.name)
            for i in range(0,self.numberOfMeasurements):
                start=cuda.Event()
                end=cuda.Event()
                start.record()
                func = kernel.module.get_function("process")
                func(self.width, self.height, self.imagesGPU, self.resultGPU, kernel.parameter, block=(16, 16, 1), grid=(int(self.width/(16/kernel.wgDivide[0])), int(self.height/(16/kernel.wgDivide[1]))), shared=0)
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
    kt.allocGPUResources()
    kt.compileKernels()
    kt.runKernels()
except Exception as e:
    print(e)
    print(traceback.format_exc())
