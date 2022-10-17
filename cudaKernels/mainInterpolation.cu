extern "C"
__global__ void process(uchar4 *inputImages, uchar4 *result, half weights[WEIGHTS_ROWS][WEIGHTS_COLS], const int2 * __restrict__  image_starts, unsigned int *atomic_counter)
{
    images.init(inputImages, result);
    int2 coords = getImgCoords();
    if(coordsOutside(coords))
        return;
    interpolateImages(weights, coords, image_starts, atomic_counter);
    }

extern "C"
__global__ void precalc_image_starts(int2 *start_images, float focus)
{
    float2 gridCenter{(GRID_COLS-1)/2.f, (GRID_ROWS-1)/2.f};
    int gridID = threadIdx.x + blockIdx.x * blockDim.x;
    int2 coord = getImageStartCoords(focus, {gridID/GRID_COLS, gridID%GRID_COLS}, gridCenter);
    start_images[gridID] = coord;
}
