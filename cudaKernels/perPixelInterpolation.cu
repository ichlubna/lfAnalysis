__device__ void interpolateImages(Images images, unsigned char *result, half *weights, half weightSum, uint2 coords, int focus)
{
    float sum[]{0,0,0,0};
    float2 gridCenter{(GRID_COLS-1)/2.f, (GRID_ROWS-1)/2.f};
    //float maxDistance = squaredDistance({0,0}, gridCenter);
    for(unsigned int y = 0; y<GRID_ROWS; y++)
        for(unsigned int x = 0; x<GRID_COLS; x++)
        {
            int2 focusedCoords = focusCoords(coords, 10, {x,y}, gridCenter);
            int gridID = getLinearID({y,x}, GRID_COLS);
            //float weight = maxDistance - squaredDistance({float(x),float(y)}, gridCenter);
            float weight{weights[gridID]};
            uchar4 pixel = images.getPixel(gridID, focusedCoords);
            float fPixel[]{float(pixel.x), float(pixel.y), float(pixel.z), float(pixel.w)};
            for(int j=0; j<4; j++)
                //sum[j] += fPixel[j]*weight;
                sum[j] = __fmaf_rn(fPixel[j], weight, sum[j]);
        }
    for(int j=0; j<4; j++)
        sum[j] /= (float)weightSum;
    uchar4 chSum{(unsigned char)sum[0], (unsigned char)sum[1], (unsigned char)sum[2], (unsigned char)sum[3]};
    images.setPixel(coords, chSum);
}
