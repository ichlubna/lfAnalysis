#include <mma.h>
#ifndef ROWS_PER_THREAD
    #define ROWS_PER_THREAD 1
#endif

using namespace nvcuda;

typedef struct {float r,g,b;} Pixel;
__device__ int2 getImgCoords()
{
    int2 coords;
    coords.x = (threadIdx.x + blockIdx.x * blockDim.x);
    coords.y = ((threadIdx.y + blockIdx.y * blockDim.y) * ROWS_PER_THREAD);
    return coords;
}

__device__ int getLinearID(int2 coords, int width)
{
    return width*coords.y + coords.x;
}

__device__ int2 focusCoords(int2 coords, int focus, int2 position, float2 center)
{
    float2 offset{center.x-position.x, center.y-position.y};
    return {__float2int_rn(focus*offset.x+coords.x), __float2int_rn(offset.y*focus+coords.y)};
}

__device__ int2 getImageStartCoords(int focus, int2 position, float2 center)
{
    float2 offset{center.x-position.x, center.y-position.y};
    return {__float2int_rn(focus*offset.x), __float2int_rn(offset.y*focus)};
}

class Images
{
    public:
        uchar4 *inData[GRID_COLS*GRID_ROWS];
        uchar4 *outData[WEIGHTS_COLS];
        int width = 0;
        int height = 0;
        __device__ Images(int w, int h) : width{w}, height{h}{};

        __device__ int2 clamp(int2 coords)
        {
            return {max(min(coords.x, IMG_WIDTH),0), max(min(coords.y, IMG_HEIGHT),0)};
        } 

        __device__ uchar4 getPixel(int imageID, int2 coords)
        {
            int2 clamped{clamp(coords)};
            int linearCoord = getLinearID(clamped, IMG_WIDTH);
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
                    data[i] = __half2int_rn(channels[i]);
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
            
        __device__ void setChannel(int imageID, int2 coords, int channelID, unsigned char value)
        {
            reinterpret_cast<unsigned char*>(outData[imageID])[(coords.y*IMG_WIDTH + coords.x)*CHANNEL_COUNT+channelID] = value;
        }

        __device__ void setPixel(int imageID, int2 coords, uchar4 pixel)
        {
            int linearCoord = getLinearID(coords, IMG_WIDTH);
            outData[imageID][linearCoord] = pixel;
        }

};

template <typename T>
__device__ static void loadWeightsSync(T *inData, T *data, int size)
{
    if(threadIdx.x < size)
    {
        int *intLocal = reinterpret_cast<int*>(data);
        int *intIn = reinterpret_cast<int*>(inData);
        intLocal[threadIdx.x] = intIn[threadIdx.x]; 
    }
    __syncthreads();
}

template <typename TT>
class Matrix
{
    public:
    __device__ Matrix(TT* inData, int inCount, int inRows, int inCols) : data{inData}, rows{inRows}, cols{inCols}, count{inCount}, matrixSize{inRows*inCols}{}; 
    __device__ TT* ptr(int id, int row, int col)
    {
        return data+linearID(id,row,col);
    }
    
    template <typename T>
    __device__ T* ptr(int id, int row, int col) 
    {
        return reinterpret_cast<T*>(ptr(id, row, col));
    }  

    __device__ TT& ref(int id, int row, int col)
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

template <typename TT>
class MemoryPartitioner
{
    public:
    __device__ MemoryPartitioner(TT *inMemory)
    {
        memory = inMemory; 
    }

    __device__ Matrix<TT> getMatrix(int count, int rows, int cols)
    {
        int size = rows*cols*count;
        TT *arr = &(memory[consumed]);
        consumed += size;
        return {arr, count, rows, cols};
    }
    private:
    TT *memory;
    unsigned int consumed{0};
};

