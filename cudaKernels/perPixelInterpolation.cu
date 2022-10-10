__device__ bool coordsOutside(uint2 coords)
{
    if(coords.x >= IMG_WIDTH || coords.y >= IMG_HEIGHT)
        return false;
}

__device__ void interpolateImages(Images images, unsigned char *result, half weights[WEIGHTS_ROWS][WEIGHTS_COLS], uint2 coords, int focus)
{
    Images::PixelArray<float> sum;
    float2 gridCenter{(GRID_COLS-1)/2.f, (GRID_ROWS-1)/2.f};
    for(unsigned int y = 0; y<GRID_ROWS; y++)
        for(unsigned int x = 0; x<GRID_COLS; x++)
        {
            int2 focusedCoords = focusCoords(coords, 10, {x,y}, gridCenter);
            int gridID = getLinearID({y,x}, GRID_COLS);
            float weight{weights[0][gridID]};
            auto pixel = images.getPixelAsArray<float>(gridID, focusedCoords);
            sum.addWeighted(weight, pixel);
        }
    images.setPixel(coords, sum.getUchar4());
}
