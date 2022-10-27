#include "generalInterpolation.cu"

__device__ bool coordsOutside(int2 coords)
{
    if(coords.x >= IMG_WIDTH || coords.y >= IMG_HEIGHT)
        return false;
}

class Pixels
{
    private:
    uchar4 data[Constants::MAT_VIEW_COUNT];
    public:
    
    __device__ void loadFromImages(int batchID, int2 coords)
    {
        int linear = images.linear(coords);
        int batchOffset = batchID*Constants::MAT_VIEW_COUNT; 
        for(int i=0; i<Constants::MAT_VIEW_COUNT; i++)
        {
            #ifdef USE_TEXTURES
            uchar4 px = images.getPixel(batchOffset+i, coords);
            #else
            uchar4 px = images.getPixel(linear + images.linear(batchOffset+i));
            #endif
            for(int j=0; j<CHANNEL_COUNT; j++)
                reinterpret_cast<unsigned char*>(data)[Constants::MAT_VIEW_COUNT*j + i] = reinterpret_cast<unsigned char*>(&px)[j];
        }
    }

    template <typename T>
    __device__ void loadFromMatrix(T *matrix, int channelID)
    {
       int offset = channelID*OUT_VIEWS_COUNT/sizeof(T);
       for(int i=0; i<OUT_VIEWS_COUNT/sizeof(T); i++)
       {
            reinterpret_cast<int*>(data)[offset+i] = reinterpret_cast<int*>(matrix)[i];
       }
    }

    __device__ uchar4 getViewPixel(int viewID)
    {
        uchar4 result;
        for(int i=0; i<CHANNEL_COUNT; i++)
            reinterpret_cast<unsigned char*>(&result)[i] =__half2uint_rn(reinterpret_cast<half*>(data)[OUT_VIEWS_COUNT*i+viewID]);
        return result;
    }

    __device__ void copyChannelsToMatrix(half *matrix, int channelID)
    { 
        int linear = channelID*Constants::MAT_VIEW_COUNT;
        constexpr int VAL_COUNT{2};
        auto dataPtr = reinterpret_cast<unsigned char*>(data);
        for(int i=0; i<Constants::MAT_VIEW_COUNT/VAL_COUNT; i++)
        {
            half2 pair{__uint2half_rn(dataPtr[linear]), __uint2half_rn(dataPtr[linear+1])};
            reinterpret_cast<half2*>(matrix)[i] = pair;
            linear+=2;
        } 
    }
};

__device__ void interpolateImages(half weights[WEIGHTS_ROWS][WEIGHTS_COLS], int2 coords, const int2 * __restrict__  image_starts, unsigned int *atomic_counter)
{
    int warpID = threadIdx.x/WARP_SIZE;
    int matrixRowID = threadIdx.x%WARP_SIZE;
   
    MemoryPartitioner<half> memoryPartitioner(localMemory);
    
    auto pixelMatrix = memoryPartitioner.getMatrix(Constants::WARP_COUNT, Constants::MAT_PX_COUNT, Constants::MAT_VIEW_COUNT);
    Indexer ID;
    ID.linearIDBase(warpID, Constants::MAT_PX_COUNT*Constants::MAT_VIEW_COUNT);
    auto localWeights = memoryPartitioner.getMatrix(1, WEIGHTS_ROWS, WEIGHTS_COLS);
    loadWeightsSync<half>(weights[0], localWeights.data, WEIGHTS_COLS*WEIGHTS_ROWS/2);  

    wmma::fragment<wmma::accumulator, 32, 8, 16, half> matResult[CHANNEL_COUNT];
    wmma::fragment<wmma::matrix_a, 32, 8, 16, half, wmma::row_major> matPixels;
   #ifdef WEIGHTS_COL_MAJOR
        // col major layout - matrices 16x8 one after each other in buffer
        #define matrix_b_dir wmma::col_major
    #else
        #define matrix_b_dir wmma::row_major
    #endif
  
    #ifdef MATRIX_LOAD_ONCE
        wmma::fragment<wmma::matrix_b, 32, 8, 16, half, matrix_b_dir> matWeights[(GRID_COLS*GRID_ROWS)/Constants::MAT_VIEW_COUNT];
    #else
        wmma::fragment<wmma::matrix_b, 32, 8, 16, half, matrix_b_dir> matWeights;
    #endif
  
    Pixels pixels;
    
    int batchCount = (GRID_COLS*GRID_ROWS)/Constants::MAT_VIEW_COUNT;
    #ifdef MATRIX_LOAD_ONCE
        for(int batchID = 0; batchID < batchCount; batchID++)
        {
            wmma::load_matrix_sync(matWeights[batchID], localWeights.ptr(batchID*OUT_VIEWS_COUNT*Constants::MAT_VIEW_COUNT), WEIGHTS_COLS);
        }
    #endif
    int originalCoordY = coords.y;
    #ifdef PERSISTENT_THREADS
    PersistentThreadsWarpBlockGetter persist_thread(atomic_counter);
    unsigned int thread_id;
    while(persist_thread.getNextId(&thread_id, matrixRowID))
    {
        coords.x = thread_id%IMG_WIDTH;
        originalCoordY = (thread_id/IMG_WIDTH) * ROWS_PER_THREAD;
    #endif
        for(int row_offset = 0; row_offset < ROWS_PER_THREAD; row_offset++)
        {
            coords.y = originalCoordY+row_offset;
            
            for(int i=0; i<CHANNEL_COUNT; i++)
                wmma::fill_fragment(matResult[i], 0.0f);
            
            for(int batchID=0; batchID<batchCount; batchID++)
            {
                pixels.loadFromImages(batchID, coords);
                #ifndef MATRIX_LOAD_ONCE
                    wmma::load_matrix_sync(matWeights, localWeights.ptr(batchID*16*8), WEIGHTS_COLS);
                #endif
                for(int channelID=0; channelID<CHANNEL_COUNT; channelID++)
                {
                    pixels.copyChannelsToMatrix(pixelMatrix.ptr(ID.linearCoordsY(matrixRowID, Constants::MAT_VIEW_COUNT)), channelID); 
                    wmma::load_matrix_sync(matPixels, pixelMatrix.ptr(ID.getBase()), Constants::MAT_VIEW_COUNT);
                    #ifdef MATRIX_LOAD_ONCE
                        wmma::mma_sync(matResult[channelID], matPixels, matWeights[batchID], matResult[channelID]);
                    #else
                        wmma::mma_sync(matResult[channelID], matPixels, matWeights, matResult[channelID]);
                    #endif
    
                    //focused TODO 
                }
            }

            for(int channelID=0; channelID<CHANNEL_COUNT; channelID++)
            {
                wmma::store_matrix_sync(pixelMatrix.ptr(ID.getBase()), matResult[channelID], OUT_VIEWS_COUNT, wmma::mem_row_major);
                pixels.loadFromMatrix<half>(pixelMatrix.ptr(ID.linearCoordsY(matrixRowID, OUT_VIEWS_COUNT)), channelID);
            }
        
            Indexer vID;
            vID.linearCoordsBase(coords, IMG_WIDTH); 
            for(int viewID = 0; viewID<OUT_VIEWS_COUNT; viewID++)
                images.setPixel(vID.linearID(viewID, IMG_WIDTH*IMG_HEIGHT), pixels.getViewPixel(viewID));
            rowSync();
        }
    #ifdef PERSISTENT_THREADS
    }
    #endif
}

#include "mainInterpolation.cu"
