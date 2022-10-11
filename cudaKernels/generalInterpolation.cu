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
        uchar4 *outData[WEIGHTS_COLS];
        int width = 0;
        int height = 0;
        __device__ Images(int w, int h) : width{w}, height{h}{};

        __device__ uchar4 getPixel(int imageID, int2 coords)
        {
            uint2 clamped{(unsigned int)max(min(coords.x, IMG_WIDTH),0), (unsigned int)max(min(coords.y, IMG_HEIGHT),0)};
            unsigned int linearCoord = getLinearID(clamped, IMG_WIDTH);
            return inData[imageID][linearCoord];
        }
 
        template <typename T>
        class PixelArray
        {
            public:
            __device__ PixelArray(){};
            __device__ PixelArray(uchar4 pixel) : channels{T(pixel.x), T(pixel.y), T(pixel.z), T(pixel.w)}{};
            static constexpr int CHANNELS_COUNT{4};
            T channels[CHANNELS_COUNT]{0,0,0,0};
            __device__ T& operator[](int index){return channels[index];}
            __device__ uchar4 getUchar4() {return {(unsigned char)channels[0], (unsigned char)channels[1], (unsigned char)channels[2], (unsigned char)channels[3]};}
            __device__ void addWeighted(T weight, PixelArray<T> value) 
            {    
                for(int j=0; j<CHANNELS_COUNT; j++)
                    //sum[j] += fPixel[j]*weight;
                    channels[j] = __fmaf_rn(value[j], weight, channels[j]);
            }
            __device__ PixelArray<T> operator/= (const T &divisor)
            {
                for(int j=0; j<CHANNELS_COUNT; j++)
                    this->channels[j] /= divisor;
                return *this;
            }
        };

        template <typename T>
        __device__ PixelArray<T> getPixelAsArray(int imageID, int2 coords)
        {
            uchar4 pixel = getPixel(imageID, coords);
            PixelArray<T> array{pixel};
            return array;
        }

        __device__ void setPixel(int imageID, uint2 coords, uchar4 pixel)
        {
            unsigned int linearCoord = getLinearID(coords, IMG_WIDTH);
            outData[imageID][linearCoord] = pixel;
        }

};

__device__ static void loadWeights(half *inData, half *data)
{
    int *intLocal = reinterpret_cast<int*>(data);
    int *intIn = reinterpret_cast<int*>(inData);
    intLocal[threadIdx.x] = intIn[threadIdx.x]; 
}
