8x8 group
#include "generalInterpolation.cu"

__device__ bool coordsOutside(int2 coords)
{
    if(coords.x >= IMG_WIDTH || coords.y >= IMG_HEIGHT)
        return false;
}

__device__ float colorDistance(Images::PixelArray<float> c1, Images::PixelArray<float> c2)
{
    return max(max(abs(c1[0]-c2[0]), abs(c1[1]-c2[1])), abs(c1[2]-c2[2]));
}

class OnlineVariance
{
    private:
    float weightSum{0};
    float m2{0};
    Images::PixelArray<float> mean{0};
    
    public:
    __device__ void add(Images::PixelArray<float> val, float weight)
    {
       weightSum += weight;
       float delta = val - mean;
       float dist = colorDistance(val, mean);
       mean.addWeighted(delta, 1.0f/weightSum);
       m2 += dist * colorDistance(val, mean);
    }
    __device__ float variance()
    {
        return m2;    
    }      
};

__device__ void interpolateImages(half weights[WEIGHTS_ROWS][WEIGHTS_COLS], int2 coords, const int2 * __restrict__  image_starts, unsigned int *atomic_counter)
{
    MemoryPartitioner<half> memoryPartitioner(localMemory);
    auto localWeights = memoryPartitioner.getMatrix(1, WEIGHTS_ROWS, WEIGHTS_COLS);
    loadWeightsSync<half>(weights[0], localWeights.data, WEIGHTS_COLS*WEIGHTS_ROWS/2);  
    Indexer weightMatIndex;

    #ifdef PERSISTENT_THREADS
    PersistentThreadsWarpBlockGetter persist_thread(atomic_counter);
    unsigned int thread_id;
    int matrixRowID = threadIdx.x%WARP_SIZE;
    while(persist_thread.getNextId(&thread_id, matrixRowID))
    {
        coords.x = thread_id%IMG_WIDTH;
        coords.y = (thread_id/IMG_WIDTH) * ROWS_PER_THREAD;
    #endif
        constexpr int FOCUS_MAX_STEPS{256};
        constexpr int FOCUS_STEP{1};
        constexpr float2 CENTER{GRID_COLS/2.0f, GRID_ROWS/2.0f};
        int originalCoordsY = coords.y;
        for(int row_offset = 0; row_offset < ROWS_PER_THREAD; row_offset++)
        {
            coords.y = originalCoordsY + row_offset;
            for(int focus = 0; focus < FOCUS_MAX_STEPS*FOCUS_STEP; focus+=FOCUS_STEP) 
            {
                Indexer pxID;
                for(int x = 0; x<GRID_COLS; x++)
                for(int y = 0; y<GRID_ROWS; y++)
                { 
                    auto focusedCoords = focusCoords(coords, focus, {x,y}, CENTER);
                    #ifdef USE_TEXTURES
                    auto pixel = images.getPixelAsArray<float>(gridID, {focusedCoords.x, focusedCords.y+row_offset});
                    #else
                    pxID.linearCoordsBase(focusedCoords, IMG_WIDTH);
                    auto pixel = images.getPixelAsArray<float>(pxID.linearID(gridID, IMG_WIDTH*IMG_HEIGHT));
                    #endif
                    for(int i=0; i<OUT_VIEWS_COUNT; i++)
                    {
                        #ifdef WEIGHTS_COL_MAJOR
                            int x = gridID%16;
                            int y = i + (OUT_VIEWS_COUNT*(gridID/16));
                        #else
                            int x = i;
                            int y = gridID;
                        #endif
TT
                        sum[i].addWeighted(localWeights.ref(weightMatIndex.linearCoords({x,y}, WEIGHTS_COLS)), pixel);
                        //sum[i].addWeighted(0, pixel);
                    }
                }
                for(int i=0; i<OUT_VIEWS_COUNT; i++)
                {
                    images.setPixel(pxID.linearID(i, IMG_WIDTH*IMG_HEIGHT), sum[i].getUchar4());
                    sum[i] = Images::PixelArray<float>();
                }
                rowSync();
            } 
        }
    #ifdef PERSISTENT_THREADS
    }
    #endif
}

#include "mainInterpolation.cu"
