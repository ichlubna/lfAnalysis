extern "C"
__global__ void process(unsigned char inputImages[GRID_COLS*GRID_ROWS][IMG_WIDTH*IMG_HEIGHT*4], unsigned char *result, half *weights, float weightSum, int parameter)
{
    Images images(IMG_WIDTH, IMG_HEIGHT);
    for(int i=0; i<GRID_COLS*GRID_ROWS; i++)
        images.inData[i] = reinterpret_cast<uchar4*>(inputImages[i]);
    images.outData = reinterpret_cast<uchar4*>(result);

    uint2 coords = getImgCoords();
    if(coords.x >= IMG_WIDTH || coords.y >= IMG_HEIGHT)
        return;

    interpolateImages(images, result, weights, (half)weightSum, coords, parameter);
    }
