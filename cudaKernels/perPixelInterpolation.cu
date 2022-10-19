#include "generalInterpolation.cu"

__device__ bool coordsOutside(int2 coords)
{
    if(coords.x >= IMG_WIDTH || coords.y >= IMG_HEIGHT)
        return false;
}

__device__ void interpolateImages(half weights[WEIGHTS_ROWS][WEIGHTS_COLS], int2 coords, const int2 * __restrict__  image_starts, unsigned int *atomic_counter)
{
    MemoryPartitioner<half> memoryPartitioner(localMemory);
    auto localWeights = memoryPartitioner.getMatrix(1, WEIGHTS_ROWS, WEIGHTS_COLS);
    loadWeightsSync<half>(weights[0], localWeights.data, WEIGHTS_COLS*WEIGHTS_ROWS/2);  
    Indexer weightMatIndex;

    Images::PixelArray<float> sum[OUT_VIEWS_COUNT];
    #ifdef PERSISTENT_THREADS
    PersistentThreadsWarpBlockGetter persist_thread(atomic_counter);
    unsigned int thread_id;
    int matrixRowID = threadIdx.x%WARP_SIZE;
    while(persist_thread.getNextId(&thread_id, matrixRowID))
    {
        coords.x = thread_id%IMG_WIDTH;
        coords.y = (thread_id/IMG_WIDTH) * ROWS_PER_THREAD;
    #endif
        for(int row_offset = 0; row_offset < ROWS_PER_THREAD; row_offset++)
        {
            Indexer pxID;
            pxID.linearCoordsBase({coords.x, coords.y+row_offset}, IMG_WIDTH);
            for(int gridID = 0; gridID<GRID_ROWS*GRID_COLS; gridID++)
            {
                //int2 focusedCoords{coords.x + image_starts[gridID].x,coords.y + image_starts[gridID].y};
                auto pixel = images.getPixelAsArray<float>(pxID.linearID(gridID, IMG_WIDTH*IMG_HEIGHT));
                for(int i=0; i<OUT_VIEWS_COUNT; i++)
                {
                    #ifdef WEIGHTS_COL_MAJOR
                        int x = gridID%16;
                        int y = i + (OUT_VIEWS_COUNT*(gridID/16));
                    #else
                        int x = i;
                        int y = gridID;
                    #endif
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
    #ifdef PERSISTENT_THREADS
    }
    #endif
}

#include "mainInterpolation.cu"
