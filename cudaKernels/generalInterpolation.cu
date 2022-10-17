#include <mma.h>
#ifndef ROWS_PER_THREAD
    #define ROWS_PER_THREAD 1
#endif

using namespace nvcuda;

extern __shared__ half localMemory[];

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
        static constexpr int IMG_SIZE{IMG_WIDTH*IMG_HEIGHT};
        uchar4 *inData;
        uchar4 *outData;
        __device__ void init(uchar4 *input, uchar4 *output)
        {
            inData = input;
            outData = output;
        };

        __device__ int2 clamp(int2 coords)
        {
            return {max(min(coords.x, IMG_WIDTH),0), max(min(coords.y, IMG_HEIGHT),0)};
        }

        __device__ int linear(int2 coords)
        {
            return coords.y*IMG_WIDTH + coords.x;
        } 

        __device__ int linear(int imageID)
        {
            return imageID*IMG_SIZE;
        }

        __device__ int linear(int2 coords, int imageID)
        {
            return linear(imageID) + linear(coords); 
        }

        __device__ uchar4 getPixel(int linearID)
        {
            return inData[linearID];
        }

        __device__ uchar4 getPixel(int imageID, int2 coords)
        {
            int2 clamped{clamp(coords)};
            return inData[linear(clamped, imageID)];
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
       
        template <typename T> 
        __device__ PixelArray<T> getPixelAsArray(int linearID)
        {
            uchar4 pixel = getPixel(linearID);
            PixelArray<T> array{pixel};
            return array;
        }
            
        __device__ void setChannel(int imageID, int2 coords, int channelID, unsigned char value)
        {
            reinterpret_cast<unsigned char*>(outData)[imageID*IMG_SIZE + (coords.y*IMG_WIDTH + coords.x)*CHANNEL_COUNT+channelID] = value;
        }

        __device__ void setPixel(int imageID, int2 coords, uchar4 pixel)
        {
            outData[linear(coords, imageID)] = pixel;
        }
        
        __device__ void setPixel(int linearID, uchar4 pixel)
        {
            outData[linearID] = pixel;
        }

};
__device__ Images images;


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

class Indexer
{
    public:
    __device__ int linearIDBase(int id, int size)
    {
        return linearCoord = id*size;
    } 
    
    __device__ int linearID(int id, int size)
    {
        return linearCoord + id*size;
    }
    
    __device__ int linearCoordsBase(int2 coords, int width)
    {
        return linearCoord = coords.y*width + coords.x;
    }

    __device__ int linearCoords(int2 coords, int width)
    {
        return linearCoord + coords.y*width + coords.x;
    }
   
    __device__ int linearCoordsY(int coordY, int width)
    {
        return linearCoord + coordY*width;
    }

    __device__ int getBase()
    {
        return linearCoord;
    }

    private:
    int linearCoord{0};
};

template <typename TT>
class Matrix
{
    public:
    __device__ Matrix(TT* inData) : data{inData}{}; 
    __device__ TT* ptr(int index)
    {
        return data+index;
    }
    
    template <typename T>
    __device__ T* ptr(int index) 
    {
        return reinterpret_cast<T*>(ptr(index));
    }  

    __device__ TT& ref(int index)
    {
        return *ptr(index);
    }
    
    template <typename T>
    __device__ T& ref(int index)
    {
        return *ptr<T>(index);
    }
 
    half *data;
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
        return {arr};
    }
    private:
    TT *memory;
    unsigned int consumed{0};
};
