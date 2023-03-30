import sys
from generalTools import lfreader as lfr
from compression import compressor as comp
from compression import pixeltools as pxtls

def main():
    lfReader = lfr.LfReader()
    compressor = comp.Compressor()
    pixelTools = pxtls.PixelTools()

    path = sys.argv[1]
    lfReader.loadDir(path)
    lfReader.setPixFmt("YCbCr")
    #values = lfReader.getPxFunction(100,100)
    values = lfReader.getRowFunction(100,1,1)
    #channels = pixelTools.splitChannels(values[0])
    channels = pixelTools.splitChannels(values)
    compressor.analyze1D(channels[0])

main()
