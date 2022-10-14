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

        __device__ uint2 clamp(int2 coords)
        {
            return {(unsigned int)max(min(coords.x, IMG_WIDTH),0), (unsigned int)max(min(coords.y, IMG_HEIGHT),0)};
        } 

        __device__ uchar4 getPixel(int imageID, int2 coords)
        {
            uint2 clamped{clamp(coords)};
            unsigned int linearCoord = getLinearID(clamped, IMG_WIDTH);
            return inData[imageID][linearCoord];
        }
 
        template <typename T>
        class PixelArray
        {
            public:
            __device__ PixelArray(){};
            __device__ PixelArray(uchar4 pixel) : channels{T(pixel.x), T(pixel.y), T(pixel.z), T(pixel.w)}{};
            T channels[CHANNEL_COUNT]{0,0,0,0};
            __device__ T& operator[](int index){return channels[index];}
          
             __device__ uchar4 getUchar4() 
            {
                uchar4 result;
                auto data = reinterpret_cast<unsigned char*>(&result);
                for(int i=0; i<CHANNEL_COUNT; i++)
                    data[i] = (unsigned char)round((float)channels[i]);
                return result;
            }
           
            __device__ void addWeighted(T weight, PixelArray<T> value) 
            {    
                for(int j=0; j<CHANNEL_COUNT; j++)
                    //sum[j] += fPixel[j]*weight;
                    channels[j] = __fmaf_rn(value[j], weight, channels[j]);
            }
            
            __device__ PixelArray<T> operator/= (const T &divisor)
            {
                for(int j=0; j<CHANNEL_COUNT; j++)
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
            
        __device__ void setChannel(int imageID, uint2 coords, int channelID, unsigned char value)
        {
            reinterpret_cast<unsigned char*>(outData[imageID])[(coords.y*IMG_WIDTH + coords.x)*CHANNEL_COUNT+channelID] = value;
        }

        __device__ void setPixel(int imageID, uint2 coords, uchar4 pixel)
        {
            unsigned int linearCoord = getLinearID(coords, IMG_WIDTH);
            outData[imageID][linearCoord] = pixel;
        }

};

__device__ static void loadWeightsSync(half *inData, half *data)
{
    if(threadIdx.x < WEIGHTS_COLS*WEIGHTS_ROWS/2)
    {
        int *intLocal = reinterpret_cast<int*>(data);
        int *intIn = reinterpret_cast<int*>(inData);
        intLocal[threadIdx.x] = intIn[threadIdx.x]; 
    }
    __syncthreads();
}

class Matrix
{
    public:
    __device__ Matrix(half* inData, int inCount, int inRows, int inCols) : data{inData}, rows{inRows}, cols{inCols}, count{inCount}, matrixSize{inRows*inCols}{}; 
    __device__ half* ptr(int id, int row, int col)
    {
        return data+linearID(id,row,col);
    }
    
    template <typename T>
    __device__ T* ptr(int id, int row, int col) 
    {
        return reinterpret_cast<T*>(ptr(id, row, col));
    }  

    __device__ half& ref(int id, int row, int col)
    {
        return *ptr(id, row, col);
    }
    
    template <typename T>
    __device__ T& ref(int id, int row, int col)
    {
        return *ptr<T>(id, row, col);
    }

    __device__ int stride() 
    {
        return cols;
    }
 
    half *data;

    private:
    int rows;
    int cols;
    int count;
    int matrixSize;
    __device__ int linearID(int id, int row, int col)
    {
        return id*matrixSize + row*cols + col;
    }
};

class MemoryPartitioner
{
    public:
    __device__ MemoryPartitioner(half *inMemory)
    {
        memory = inMemory; 
    }

    __device__ Matrix getMatrix(int count, int rows, int cols)
    {
        int size = rows*cols*count;
        half *arr = &(memory[consumed]);
        consumed += size;
        return {arr, count, rows, cols};
    }
    private:
    half *memory;
    unsigned int consumed{0};
};

