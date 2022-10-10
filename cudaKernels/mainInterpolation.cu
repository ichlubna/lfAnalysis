extern "C"
__global__ void process(unsigned char inputImages[GRID_COLS*GRID_ROWS][IMG_WIDTH*IMG_HEIGHT*4], unsigned char *result, half weights[WEIGHTS_ROWS][WEIGHTS_COLS], half weightSums[WEIGHTS_ROWS], int parameter)
{
    Images images(IMG_WIDTH, IMG_HEIGHT,result);
    for(int i=0; i<GRID_COLS*GRID_ROWS; i++)
        images.inData[i] = reinterpret_cast<uchar4*>(inputImages[i]);
    images.outData = reinterpret_cast<uchar4*>(result);

    uint2 coords = getImgCoords();
    if(coordsOutside(coords))
        return;

    interpolateImages(images, result, weights, weightSums, coords, parameter);
    }
