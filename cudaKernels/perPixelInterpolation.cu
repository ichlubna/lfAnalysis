__device__ bool coordsOutside(int2 coords)
{
    if(coords.x >= IMG_WIDTH || coords.y >= IMG_HEIGHT)
        return false;
}

__device__ void interpolateImages(Images images, half weights[WEIGHTS_ROWS][WEIGHTS_COLS], int2 coords, const int2 * __restrict__  image_starts, unsigned int *atomic_counter)
{
    extern __shared__ half localMemory[];
    MemoryPartitioner memoryPartitioner(localMemory);
    auto localWeights = memoryPartitioner.getMatrix(1, WEIGHTS_ROWS, WEIGHTS_COLS);
    loadWeightsSync<half>(weights[0], localWeights.data, WEIGHTS_COLS*WEIGHTS_ROWS/2);  

    Images::PixelArray<float> sum[OUT_VIEWS_COUNT];
    unsigned int mat_size = IMG_WIDTH*IMG_HEIGHT;
    uchar4 *in_char4_ptr = images.inData[0];
    uchar4 *out_char4_ptr = images.outData[0];
    for(int row_offset = 0; row_offset < ROWS_PER_THREAD; row_offset++)
    {
        int coord = coords.x + (coords.y + row_offset) * IMG_WIDTH;
        for(int gridID = 0; gridID<GRID_ROWS*GRID_COLS; gridID++)
        {
            //int2 focusedCoords{coords.x + image_starts[gridID].x,coords.y + image_starts[gridID].y};
            //auto pixel = images.getPixelAsArray<float>(gridID, focusedCoords);
            Images::PixelArray<float> pixel(in_char4_ptr[coord + gridID * mat_size]);
            for(int i=0; i<OUT_VIEWS_COUNT; i++)
            {
                #ifdef WEIGHTS_COL_MAJOR
                    int x = gridID%16;
                    int y = i + (OUT_VIEWS_COUNT*(gridID/16));
                #else
                    int x = i;
                    int y = gridID;
                #endif
                sum[i].addWeighted(localWeights.ref(0, y, x), pixel);
            }
        }
        for(int i=0; i<OUT_VIEWS_COUNT; i++)
        {
            out_char4_ptr[coord + i * mat_size] = sum[i].getUchar4();
            sum[i] = Images::PixelArray<float>();
        }
    }
}
