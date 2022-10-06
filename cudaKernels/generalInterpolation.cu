#include <mma.h>
using namespace nvcuda;

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

__device__ int2 focusCoords(uint2 coords, int focus, uint2 position, float2 center)
{
    float2 offset{center.x-position.x, center.y-position.y};
    return {__float2int_rn(focus*offset.x+coords.x), __float2int_rn(offset.y*focus+coords.y)};
}

class Images
{
    public:
        uchar4 *inData[GRID_COLS*GRID_ROWS];
        uchar4 *outData;
        int width = 0;
        int height = 0;
        __device__ Images(int w, int h) : width{w}, height{h} {};

        __device__ uchar4 getPixel(int imageID, int2 coords)
        {
            uint2 clamped{(unsigned int)max(min(coords.x, IMG_WIDTH),0), (unsigned int)max(min(coords.y, IMG_HEIGHT),0)};
            unsigned int linearCoord = getLinearID(clamped, IMG_WIDTH);
            return inData[imageID][linearCoord];
        }

        __device__ void setPixel(uint2 coords, uchar4 pixel)
        {
            unsigned int linearCoord = getLinearID(coords, IMG_WIDTH);
            outData[linearCoord] = pixel;
        }
};

