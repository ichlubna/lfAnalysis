__device__ bool coordsOutside(uint2 coords)
{
    constexpr unsigned int PX_PER_WARP{8};
    if(coords.x >= IMG_WIDTH*PX_PER_WARP || coords.y >= IMG_HEIGHT)
        return false;
}

__device__ void interpolateImages(Images images, unsigned char *result, half weights[WEIGHTS_ROWS][WEIGHTS_COLS], uint2 coords, int focus)
{
    constexpr int MAT_PX_COUNT{8}; 
    constexpr int OUT_VIEWS_COUNT{8}; 
    constexpr int MAT_VIEW_COUNT{16};
    constexpr int CHANNELS{4};

    int warpID = threadIdx.x/WARP_SIZE;
    uint2 pxCoords{coords.x/CHANNELS, coords.y};
    int channelID = threadIdx.x%CHANNELS;
    //int matrixRowID = CHANNELS*((int)(coords.x%WARP_SIZE)/CHANNELS) + channelID;
    int matrixRowID = threadIdx.x%WARP_SIZE;//coords.x%WARP_SIZE;
    float2 gridCenter{(GRID_COLS-1)/2.f, (GRID_ROWS-1)/2.f};

    __shared__ half pixelMatrix[256/WARP_SIZE][MAT_PX_COUNT*CHANNELS][MAT_VIEW_COUNT];
 
    __shared__ half localWeights[WEIGHTS_ROWS][WEIGHTS_COLS];
    int *intLocal = reinterpret_cast<int*>(localWeights[0]);
    int *intIn = reinterpret_cast<int*>(weights[0]);
    intLocal[threadIdx.x] = intIn[threadIdx.x]; 
     
    __syncthreads();

    wmma::fragment<wmma::accumulator, 32, 8, 16, half> matResult;
    wmma::fill_fragment(matResult, 0.0f);
    wmma::fragment<wmma::matrix_a, 32, 8, 16, half, wmma::row_major> matPixels;
    wmma::fragment<wmma::matrix_b, 32, 8, 16, half, wmma::row_major> matWeights;

    int batchCount = (GRID_COLS*GRID_ROWS)/MAT_VIEW_COUNT;
    for(int i=0; i<batchCount; i++)
    {
        wmma::load_matrix_sync(matWeights, localWeights[i*MAT_VIEW_COUNT], OUT_VIEWS_COUNT);

        for(int j=0; j<MAT_VIEW_COUNT; j++)
        {
            int gridID = i*MAT_VIEW_COUNT+j; 
            int2 focusedCoords = focusCoords(pxCoords, 10, {(unsigned int)gridID/GRID_COLS, (unsigned int)gridID%GRID_COLS}, gridCenter);
            auto pixel = images.getPixelAsArray<half>(gridID, focusedCoords);
            pixelMatrix[warpID][matrixRowID][j] = pixel[channelID]; 
            //for(int j=0; j<4; j++)
            //      sum[j] = matAcc.x[0];
        }
        wmma::load_matrix_sync(matPixels, pixelMatrix[warpID][0], MAT_VIEW_COUNT);
        wmma::mma_sync(matResult, matPixels, matWeights, matResult);
    }
    
    __shared__ half resultMatrix[256/WARP_SIZE][MAT_PX_COUNT*CHANNELS][OUT_VIEWS_COUNT];
    wmma::store_matrix_sync(resultMatrix[warpID][0], matResult, OUT_VIEWS_COUNT, wmma::mem_row_major);
    
    //for all views udelat 
    images.rawOutData[coords.y*IMG_WIDTH*CHANNELS + coords.x] = round((float)resultMatrix[warpID][matrixRowID][0]);
}
