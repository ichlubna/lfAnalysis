extern "C"
__global__ void process(uchar4 inputImages[GRID_COLS*GRID_ROWS][IMG_WIDTH*IMG_HEIGHT], uchar4 result[WEIGHTS_COLS][IMG_WIDTH*IMG_HEIGHT], half weights[WEIGHTS_ROWS][WEIGHTS_COLS], int parameter)
{
    Images images(IMG_WIDTH, IMG_HEIGHT);
    for(int i=0; i<GRID_COLS*GRID_ROWS; i++)
        images.inData[i] = inputImages[i];
    for(int i=0; i<WEIGHTS_COLS; i++)
        images.outData[i] = result[i];

    int2 coords = getImgCoords();
    if(coordsOutside(coords))
        return;

    interpolateImages(images, weights, coords, parameter);
    }
