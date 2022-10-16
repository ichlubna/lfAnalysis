extern "C"
__global__ void process(uchar4 inputImages[GRID_COLS*GRID_ROWS][IMG_WIDTH*IMG_HEIGHT], uchar4 result[8][IMG_WIDTH*IMG_HEIGHT], half weights[WEIGHTS_ROWS][WEIGHTS_COLS], const int2 * __restrict__  image_starts, unsigned int *atomic_counter)
{
    Images images(IMG_WIDTH, IMG_HEIGHT);
    for(int i=0; i<GRID_COLS*GRID_ROWS; i++)
        images.inData[i] = inputImages[i];
    for(int i=0; i<OUT_VIEWS_COUNT; i++)
        images.outData[i] = result[i];

    int2 coords = getImgCoords();
    if(coordsOutside(coords))
        return;
    /*if (coords.x == 0 && coords.y == 0)
    {
        printf("%d %d \n", (int)WEIGHTS_ROWS, (int)WEIGHTS_COLS);
    }*/
    interpolateImages(images, weights, coords, image_starts, atomic_counter);
    }

extern "C"
__global__ void precalc_image_starts(int2 *start_images, float focus)
{
    float2 gridCenter{(GRID_COLS-1)/2.f, (GRID_ROWS-1)/2.f};
    int gridID = threadIdx.x + blockIdx.x * blockDim.x;
    int2 coord = getImageStartCoords(focus, {gridID/GRID_COLS, gridID%GRID_COLS}, gridCenter);
    start_images[gridID] = coord;
}