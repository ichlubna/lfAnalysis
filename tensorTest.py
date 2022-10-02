import lfreader as lfr
import evaluator as eva
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy
import cv2
import sys

lfReader = lfr.LfReader()
lfReader.setPixFmt("RGBA")
path = sys.argv[1]
lfReader.loadDir(path)
colsRows = lfReader.getColsRows()
leftTopX = int(colsRows[0]/2)-1
leftTopY = int(colsRows[1]/2)-1
width, height = lfReader.getResolution()
width = numpy.int32(width)
height = numpy.int32(height)
depth = numpy.int32(4)
size = int(width*height*depth)

images = bytes()
imageCount = 4

for i in range(0,imageCount):
    img = lfReader.openImage(leftTopY+i%2, int(leftTopX+i/2))
    imgBytes = img.tobytes()
    images = images + imgBytes

imagesGPU = cuda.mem_alloc(imageCount*size)
cuda.memcpy_htod(imagesGPU, images)

result = numpy.zeros((height, width, depth), numpy.uint8)
resultGPU = cuda.mem_alloc(int(size))

kernelConstants = ["-DIMG_SIZE="+str(size), "-DIMG_COUNT="+str(imageCount)]

kernelSourceGeneral = """
typedef struct {float r,g,b;} Pixel;
__device__ uint2 getImgCoords()
{
    uint2 coords;
    coords.x = threadIdx.x + blockIdx.x * blockDim.x;
    coords.y = threadIdx.y + blockIdx.y * blockDim.y;
    return coords;
}

__device__ unsigned int getLinearID(unsigned int width, uint2 coords)
{
    const unsigned int depth = 4;
    return depth*(width*coords.y + coords.x);

}

__device__ uchar4 getPixel(unsigned char *image, int id)
{
    uchar4 pixel;
    pixel.x = image[id];
    pixel.y = image[id+1];
    pixel.z = image[id+2];
    pixel.w = image[id+3];
    return pixel;
}

__device__ void setPixel(unsigned char *image, int id, uchar4 pixel)
{
    image[id] = pixel.x;
    image[id+1] = pixel.y;
    image[id+2] = pixel.z;
    image[id+3] = pixel.w;
}
"""

kernelSourcePerPixel = """
__device__ void interpolateImages(unsigned char images[IMG_COUNT][IMG_SIZE], int id, unsigned char *result)
{
    float weights[]{0.2f, 0.5f, 0.1f, 0.2f};
    float sum[]{0,0,0,0};
    for(int i = 0; i<IMG_COUNT; i++)
    {
        uchar4 pixel = getPixel(images[i], id);
        float fPixel[]{float(pixel.x), float(pixel.y), float(pixel.z), float(pixel.w)};
        for(int j=0; j<4; j++)
            //sum[j] += fPixel[j] * weights[i];
            sum[j] = __fmaf_rn(fPixel[j], weights[i], sum[j]);
    }
    uchar4 chSum{sum[0], sum[1], sum[2], sum[3]};
    setPixel(result, id, chSum);
}

__global__ void process( int width, int height,
unsigned char images[IMG_COUNT][IMG_SIZE], unsigned char *result )
  {
    unsigned char* leftTop = images[0];
    uint2 coords = getImgCoords();
    if(coords.x >= width || coords.y >= height)
        return;
    int id = getLinearID(width, coords);
    setPixel(result, id, getPixel(leftTop, id));
    interpolateImages(images, id, result);

  }
"""

perPixelKernel = SourceModule(kernelSourceGeneral+kernelSourcePerPixel, options=kernelConstants)

kernels = [("Per pixel", perPixelKernel)]

for kernel in kernels:
    print(kernel[0])

    start=cuda.Event()
    end=cuda.Event()
    start.record()
    func = kernel[1].get_function("process")
    func(width, height, imagesGPU, resultGPU, block=(16, 16, 1), grid=(int(width/16), int(height/16)), shared=0)
    end.record()
    end.synchronize()
    print("Time: "+str(start.time_till(end))+" ms")

    cuda.memcpy_dtoh(result, resultGPU)
    result = result.astype(numpy.uint8)
    resultImage = Image.frombytes("RGBA", (width, height), result)
    resultImage.save("./distorted/test.png")
    evaluator = eva.Evaluator()
    print(evaluator.metrics("./original", "./distorted"))
