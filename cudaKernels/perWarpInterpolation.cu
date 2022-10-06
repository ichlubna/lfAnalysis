// 8x32 32x16 8x16 colxrow
//32*r 16*w first column result
//preloaded weights load_matrix_sync

__device__ void interpolateImages(Images images, unsigned char *result, half *weights, half weightSum, uint2 coords, int focus)
{
    constexpr int WARP_SIZE{32};
    float2 gridCenter{GRID_COLS/2.f, GRID_ROWS/2.f};
    uint2 warpCoords{coords.x/WARP_SIZE, coords.y};
    int threadID = threadIdx.x;
    //half sum[]{0,0,0,0};
    float sum[]{0,0,0,0};
    for(int i=0; i<2; i++)
    {
        int linearID = i*WARP_SIZE + threadID;
        unsigned int x = linearID % GRID_COLS;
        unsigned int y = linearID / GRID_ROWS;
        int2 focusedCoords = focusCoords(coords, 10, {x,y}, gridCenter);
        uchar4 pixel = images.getPixel(linearID, focusedCoords);
        //half hPixel[]{half(pixel.x), half(pixel.y), half(pixel.z), half(pixel.w)};
        //half weight{weights[linearID]};
        float hPixel[]{float(pixel.x), float(pixel.y), float(pixel.z), float(pixel.w)};
        float weight{(float)weights[linearID]};
        for(int j=0; j<4; j++)
            sum[j] += hPixel[j]*weight;
    }
    for(int i=1; i<=16; i*=2)
        for(int j=0; j<4; j++)
            sum[j] += __shfl_down_sync(0xffffffff, sum[j], i);

    if(threadID == 0)
    {
        for(int j=0; j<4; j++)
            sum[j] /= (float)weightSum;
        //uchar4 chSum{(unsigned char)(int)sum[0], (unsigned char)(int)sum[1], (unsigned char)(int)sum[2], (unsigned char)(int)sum[3]};
        uchar4 chSum{(unsigned char)sum[0], (unsigned char)sum[1], (unsigned char)sum[2], (unsigned char)sum[3]};
        images.setPixel(coords, chSum);
    }
/*
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> matA;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> matB;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> matAcc;

            wmma::fill_fragment(matAcc, 0.0f);
            wmma::fill_fragment(matA, 0.0f);
            wmma::fill_fragment(matB, 0.0f);
            wmma::mma_sync(matAcc, matA, matB, matAcc);
            for(int j=0; j<4; j++)
                sum[j] = matAcc.x[0];
*/
}
