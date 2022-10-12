//(8*8*4*16*2 + 64*8*2 + 8*8*4*8*2)/1000
//(8*8*4*16*2 + 64*8*2 + 8*8*4*8*2)/1000
__device__ bool coordsOutside(uint2 coords)
{
    constexpr unsigned int PX_PER_WARP{8};
    if(coords.x >= IMG_WIDTH*PX_PER_WARP || coords.y >= IMG_HEIGHT)
        return false;
}

__device__ void interpolateImages(Images images, half weights[WEIGHTS_ROWS][WEIGHTS_COLS], uint2 coords, int focus)
{
    constexpr int MAT_PX_COUNT{8};
    constexpr int WARP_COUNT{8}; 
    constexpr int OUT_VIEWS_COUNT{WEIGHTS_COLS}; 
    constexpr int MAT_VIEW_COUNT{16};
    constexpr int CHANNELS{4};
    constexpr int MATS_PER_WARP{1};

    int warpID = threadIdx.x/WARP_SIZE;
    uint2 pxCoords{coords.x/CHANNELS, coords.y};
    int channelID = threadIdx.x%CHANNELS;
    //int matrixRowID = CHANNELS*((int)(coords.x%WARP_SIZE)/CHANNELS) + channelID;
    int matrixRowID = threadIdx.x%WARP_SIZE;//coords.x%WARP_SIZE;
    float2 gridCenter{(GRID_COLS-1)/2.f, (GRID_ROWS-1)/2.f};

    extern __shared__ half localMemory[];
    MemoryPartitioner memoryPartitioner(localMemory);
    
    auto pixelMatrix = memoryPartitioner.getMatrix(MATS_PER_WARP*WARP_COUNT, MAT_PX_COUNT*CHANNELS, MAT_VIEW_COUNT); 
    auto resultMatrix = memoryPartitioner.getMatrix(MATS_PER_WARP*WARP_COUNT, MAT_PX_COUNT*CHANNELS, OUT_VIEWS_COUNT);
    auto localWeights = memoryPartitioner.getMatrix(1, WEIGHTS_ROWS, WEIGHTS_COLS);
    loadWeightsSync(weights[0], localWeights.data);  

    wmma::fragment<wmma::accumulator, 32, 8, 16, half> matResult;
    wmma::fill_fragment(matResult, 0.0f);
    wmma::fragment<wmma::matrix_a, 32, 8, 16, half, wmma::row_major> matPixels;
    wmma::fragment<wmma::matrix_b, 32, 8, 16, half, wmma::row_major> matWeights;

    int batchCount = (GRID_COLS*GRID_ROWS)/MAT_VIEW_COUNT;
    for(int i=0; i<batchCount; i++)
    {
        wmma::load_matrix_sync(matWeights, localWeights.ptr(0, i*MAT_VIEW_COUNT, 0), OUT_VIEWS_COUNT);

        for(int j=0; j<MAT_VIEW_COUNT; j++)
        {
            int gridID = i*MAT_VIEW_COUNT+j; 
            int2 focusedCoords = focusCoords(pxCoords, 10, {(unsigned int)gridID/GRID_COLS, (unsigned int)gridID%GRID_COLS}, gridCenter);
            auto pixel = images.getPixelAsArray<half>(gridID, focusedCoords);
            pixelMatrix(warpID, matrixRowID, j) = pixel[channelID];
            //for(int j=0; j<4; j++)
            //      sum[j] = matAcc.x[0];
        }
        wmma::load_matrix_sync(matPixels, pixelMatrix.ptr(warpID, 0, 0), MAT_VIEW_COUNT);
        wmma::mma_sync(matResult, matPixels, matWeights, matResult);
    }
    
    wmma::store_matrix_sync(resultMatrix.ptr(warpID, 0, 0), matResult, OUT_VIEWS_COUNT, wmma::mem_row_major);
    
    for(int i=0; i<OUT_VIEWS_COUNT; i++) 
        reinterpret_cast<unsigned char*>(images.outData[i])[coords.y*IMG_WIDTH*CHANNELS + coords.x] = round((float)resultMatrix(warpID, matrixRowID, i));
}
