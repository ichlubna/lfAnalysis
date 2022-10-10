// 1 warp načte 32 px
// matce rows x cols A = 32*16 + B = 16*8 = C = 32*8
// A 8 pixelů z 16 pohledů
// B 16 vah pro 8 snímků
// warp 4x (pro všech 32 - 4*8) udělá  AB + AB + AB + AB = 8 pixelů pro 8 snímků
// 4 váhovací matice po 16 pohledech a 8 snímcích, 4 matice po 8 pxelech pro 16 pohledů
// váhovací matice zůstávají takže jen 4*4 matic po 8 pxelech
// pokud 32 pixelů a načítat po 16 views 2* 32*16 *4 * 8 + 8*32*2*4*8 rtx 4090
// pokud po 16 pixelech 2* 32*16 *2 *8 + 8*32*2*2*8

__device__ bool coordsOutside(uint2 coords)
{
    constexpr unsigned int PX_PER_WARP{8};
    if(coords.x >= IMG_WIDTH*PX_PER_WARP || coords.y >= IMG_HEIGHT)
        return false;
}

__device__ void interpolateImages(Images images, unsigned char *result, half weights[WEIGHTS_ROWS][WEIGHTS_COLS], half weightSums[WEIGHTS_ROWS], uint2 coords, int focus)
{
    constexpr int MAT_PX_COUNT{8}; 
    constexpr int MAT_VIEW_COUNT{16};
    constexpr int CHANNELS{4};
    __shared__ half pixelMatrix[256/WARP_SIZE][MAT_PX_COUNT*CHANNELS][MAT_VIEW_COUNT];
    int warpID = (blockIdx.x*blockDim.y)/WARP_SIZE;
    uint2 pxCoords{coords.x/CHANNELS, coords.y};
    int channelID = coords.x%CHANNELS;
    //int matrixRowID = CHANNELS*((int)(coords.x%WARP_SIZE)/CHANNELS) + channelID;
    int matrixRowID =coords.x%WARP_SIZE;
    float2 gridCenter{(GRID_COLS-1)/2.f, (GRID_ROWS-1)/2.f};
     
    wmma::fragment<wmma::accumulator, 32, 8, 16, half> matResult;
    wmma::fill_fragment(matResult, 0.0f);
    wmma::fragment<wmma::matrix_a, 32, 8, 16, half, wmma::row_major> matPixels;
    wmma::fragment<wmma::matrix_b, 32, 8, 16, half, wmma::row_major> matWeights;

    int batchCount = __float2int_ru((float)(GRID_ROWS*GRID_ROWS)/MAT_VIEW_COUNT);
    for(int i=0; i<batchCount; i++)
    {
        wmma::load_matrix_sync(matWeights, weights[i*MAT_VIEW_COUNT], MAT_PX_COUNT);

        for(int j=0; j<MAT_VIEW_COUNT; j++)
        {
            int gridID = i*MAT_VIEW_COUNT+j; 
            int2 focusedCoords = focusCoords(pxCoords, 10, {gridID%GRID_COLS, gridID/GRID_COLS}, gridCenter);
            auto pixel = images.getPixelAsArray<float>(gridID, focusedCoords);
            pixelMatrix[warpID][matrixRowID][gridID] = pixel[channelID]; 
        //ZKUSIT BEZ SHARED
        //for(int j=0; j<4; j++)
          //      sum[j] = matAcc.x[0];
        }
        wmma::load_matrix_sync(matPixels, pixelMatrix[warpID][0], MAT_VIEW_COUNT);
        wmma::mma_sync(matResult, matPixels, matWeights, matResult);
    }
    
    __shared__ half resultMatrix[256/WARP_SIZE][MAT_PX_COUNT*CHANNELS][MAT_PX_COUNT];
    wmma::store_matrix_sync(resultMatrix[warpID][0], matResult, 0, wmma::mem_row_major);
    
    //for all views udelat 
    images.rawOutData[coords.y*IMG_WIDTH*CHANNELS + coords.x] = (unsigned int)(resultMatrix[warpID][matrixRowID][0]/weightSums[0]); 
}
