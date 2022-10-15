__device__ bool coordsOutside(int2 coords)
{
    if(coords.x >= IMG_WIDTH || coords.y >= IMG_HEIGHT)
        return false;
}

__device__ void interpolateImages(Images images, half weights[WEIGHTS_ROWS][WEIGHTS_COLS], int2 coords, int focus)
{
    extern __shared__ half localMemory[];
    MemoryPartitioner<half> memoryPartitioner(localMemory);
    auto localWeights = memoryPartitioner.getMatrix(1, WEIGHTS_ROWS, WEIGHTS_COLS);
    loadWeightsSync<half>(weights[0], localWeights.data, WEIGHTS_COLS*WEIGHTS_ROWS/2); 

    Images::PixelArray<float> sum[WEIGHTS_COLS];
    float2 gridCenter{(GRID_COLS-1)/2.f, (GRID_ROWS-1)/2.f};
    for(int y = 0; y<GRID_ROWS; y++)
        for(int x = 0; x<GRID_COLS; x++)
        {
            int2 focusedCoords = focusCoords(coords, focus, {x,y}, gridCenter);
            int gridID = getLinearID({y,x}, GRID_COLS);
            auto pixel = images.getPixelAsArray<float>(gridID, focusedCoords);
            for(int i=0; i<WEIGHTS_COLS; i++)
                sum[i].addWeighted(localWeights.ref(0, gridID, i), pixel);
        }
    for(int i=0; i<WEIGHTS_COLS; i++)
        images.setPixel(i, coords, sum[i].getUchar4());
}
