__device__ bool coordsOutside(int2 coords)
{
    constexpr int PX_PER_WARP{8};
    if(coords.x >= IMG_WIDTH*4 || coords.y >= IMG_HEIGHT)
        return false;
}

__device__ void interpolateImages(Images images, half weights[WEIGHTS_ROWS][WEIGHTS_COLS], int2 coords, const int2 * __restrict__  image_starts, unsigned int *atomic_counter)
{
    
    constexpr int MAT_PX_COUNT{8};
    constexpr int WARP_COUNT{8}; 
    constexpr int MAT_VIEW_COUNT{16};
    

    int warpID = threadIdx.x/WARP_SIZE;
    int2 pxCoords{coords.x/CHANNEL_COUNT, coords.y};
    int channelID = threadIdx.x%CHANNEL_COUNT;
    int matrixRowID = threadIdx.x%WARP_SIZE;//coords.x%WARP_SIZE;

    #ifdef PERSISTENT_THREADS
    unsigned int blocks_count = IMG_WIDTH*IMG_HEIGHT*4/WARP_SIZE;
    unsigned int blocks_per_row = IMG_WIDTH*4/WARP_SIZE;
    unsigned int act_block = 0;
    unsigned int row_offset = 0;
    if(matrixRowID == 0)
    {
        act_block = atomicAdd(atomic_counter,1);
    }
    act_block = __shfl_sync(0xffffffff,act_block,0);
    #endif

    extern __shared__ half localMemory[];
    MemoryPartitioner memoryPartitioner(localMemory);
   
    auto pixelMatrix = memoryPartitioner.getMatrix(WARP_COUNT, MAT_PX_COUNT*CHANNEL_COUNT, MAT_VIEW_COUNT); 
    auto resultMatrix = memoryPartitioner.getMatrix(WARP_COUNT, MAT_PX_COUNT*CHANNEL_COUNT, OUT_VIEWS_COUNT);
    auto localWeights = memoryPartitioner.getMatrix(1, WEIGHTS_ROWS, WEIGHTS_COLS);
    loadWeightsSync<half>(weights[0], localWeights.data, WEIGHTS_COLS*WEIGHTS_ROWS/2);  

    if (act_block >= blocks_count) return;

    wmma::fragment<wmma::accumulator, 32, 8, 16, half> matResult;
    //wmma::fill_fragment(matResult, 0.0f);
    wmma::fragment<wmma::matrix_a, 32, 8, 16, half, wmma::row_major> matPixels;
    #ifdef WEIGHTS_COL_MAJOR
        // col major layout - matrices 16x8 one after each other in buffer
        #define matrix_b_dir wmma::col_major
        constexpr int stride_y = 8;
    #else
        #define matrix_b_dir wmma::row_major
        constexpr int stride_y = MAT_VIEW_COUNT;
    #endif
    
    #ifdef MATRIX_LOAD_ONCE
        wmma::fragment<wmma::matrix_b, 32, 8, 16, half, matrix_b_dir> matWeights[(GRID_COLS*GRID_ROWS)/MAT_VIEW_COUNT];
    #else
        wmma::fragment<wmma::matrix_b, 32, 8, 16, half, matrix_b_dir> matWeights;
    #endif

    
    int batchCount = (GRID_COLS*GRID_ROWS)/MAT_VIEW_COUNT;
    #ifdef MATRIX_LOAD_ONCE
        for(int batchID = 0; batchID < batchCount; batchID++)
        {
            wmma::load_matrix_sync(matWeights[batchID], localWeights.ptr<half>(0, batchID * stride_y, 0), localWeights.stride());
        }
    #endif
    unsigned char *in_char_ptr = (unsigned char *)(images.inData[0]);
    unsigned char *out_char_ptr = (unsigned char *)(images.outData[0]);
    unsigned int mat_size = IMG_WIDTH*IMG_HEIGHT*4;
    #ifdef PERSISTENT_THREADS
    while(act_block < blocks_count){coords.x = (act_block%blocks_per_row) * WARP_SIZE + matrixRowID; coords.y = act_block/blocks_per_row;
    #else
    for(int row_offset = 0; row_offset < ROWS_PER_THREAD; row_offset++){
    #endif
        int coord = coords.x + (coords.y + row_offset) * IMG_WIDTH * 4;

        wmma::fill_fragment(matResult, 0.0f);
        
        for(int batchID=0; batchID < batchCount; batchID++)
        {
            #ifndef MATRIX_LOAD_ONCE
                wmma::load_matrix_sync(matWeights, localWeights.ptr<half>(0, batchID * stride_y, 0), localWeights.stride());
            #endif
            for(int viewID=0; viewID<MAT_VIEW_COUNT; viewID+=2)
            {
                int gridID = batchID*MAT_VIEW_COUNT+viewID; 
                half2 channelPair{(half)in_char_ptr[coord + gridID * mat_size],(half)in_char_ptr[coord + (gridID + 1) * mat_size]};
                pixelMatrix.ref<half2>(warpID, matrixRowID, viewID) = channelPair;
            }
            wmma::load_matrix_sync(matPixels, pixelMatrix.ptr(warpID, 0, 0), pixelMatrix.stride());
            #ifdef MATRIX_LOAD_ONCE
                wmma::mma_sync(matResult, matPixels, matWeights[batchID], matResult);
            #else
                wmma::mma_sync(matResult, matPixels, matWeights, matResult);
            #endif
        }
        wmma::store_matrix_sync(resultMatrix.ptr(warpID, 0, 0), matResult, OUT_VIEWS_COUNT, wmma::mem_row_major);
        for(int i = 0; i< OUT_VIEWS_COUNT; i++)
            out_char_ptr[coord + mat_size * i] = __half2int_rn(resultMatrix.ref(warpID, matrixRowID, i));
        #ifdef PERSISTENT_THREADS
        if(matrixRowID == 0)
        {
            act_block = atomicAdd(atomic_counter,1);
        }
        act_block = __shfl_sync(0xffffffff,act_block,0);
        #endif
    }
}
