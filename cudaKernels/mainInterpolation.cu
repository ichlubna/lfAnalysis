extern "C"
__global__ void process(unsigned char inputImages[GRID_COLS*GRID_ROWS][IMG_WIDTH*IMG_HEIGHT*4], unsigned char result[WEIGHTS_COLS][IMG_WIDTH*IMG_HEIGHT*4], half weights[WEIGHTS_ROWS][WEIGHTS_COLS], int parameter)
{
    Images images(IMG_WIDTH, IMG_HEIGHT);
    for(int i=0; i<GRID_COLS*GRID_ROWS; i++)
        images.inData[i] = reinterpret_cast<uchar4*>(inputImages[i]);
    for(int i=0; i<WEIGHTS_COLS; i++)
        images.outData[i] = reinterpret_cast<uchar4*>(result[i]);

    uint2 coords = getImgCoords();
    if(coordsOutside(coords))
        return;

    interpolateImages(images, weights, coords, parameter);
    }
