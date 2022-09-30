import sys
import lfreader as lfr
import compressor as comp
import pixeltools as pxtls

def main():
    lfReader = lfr.LfReader()
    compressor = comp.Compressor()
    pixelTools = pxtls.PixelTools()

    path = sys.argv[1]
    lfReader.loadDir(path)
    lfReader.setPixFmt("YCbCr")
    values = lfReader.getPxFunction(100,100)
    channels = pixelTools.splitChannels(values[0])
    compressor.analyze1D(channels[0])

main()
