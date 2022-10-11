__device__ bool coordsOutside(uint2 coords)
{
    if(coords.x >= IMG_WIDTH || coords.y >= IMG_HEIGHT)
        return false;
}

__device__ void interpolateImages(Images images, half weights[WEIGHTS_ROWS][WEIGHTS_COLS], uint2 coords, int focus)
{
    __shared__ half localWeights[WEIGHTS_ROWS][WEIGHTS_COLS];
    loadWeights(weights[0], localWeights[0]); 

    Images::PixelArray<float> sum[WEIGHTS_COLS];
    float2 gridCenter{(GRID_COLS-1)/2.f, (GRID_ROWS-1)/2.f};
    for(unsigned int y = 0; y<GRID_ROWS; y++)
        for(unsigned int x = 0; x<GRID_COLS; x++)
        {
            int2 focusedCoords = focusCoords(coords, 10, {x,y}, gridCenter);
            int gridID = getLinearID({y,x}, GRID_COLS);
            auto pixel = images.getPixelAsArray<float>(gridID, focusedCoords);
            for(int i=0; i<WEIGHTS_COLS; i++)
                sum[i].addWeighted(localWeights[gridID][i], pixel);
        }
    for(int i=0; i<WEIGHTS_COLS; i++)
        images.setPixel(i, coords, sum[i].getUchar4());
}
