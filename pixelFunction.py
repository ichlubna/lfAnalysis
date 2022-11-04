from generalTools import lfreader as lfr
import numpy
from matplotlib import pyplot as plt
import sys
import os
import traceback

class PixelAnalyzer:
    cols = 0
    rows = 0
    lfReader = None
    channels = 3
    pxFunction = [[], [], []] #RGB

    def loadInput(self):
        lfReader = lfr.LfReader()
        lfReader.setPixFmt("RGB")
        path = sys.argv[1]
        lfReader.loadDir(path)
        colsRows = lfReader.getColsRows()
        self.cols = colsRows[0]
        self.rows = colsRows[1]
        width, height = lfReader.getResolution()
        self.lfReader = lfReader

    def getPixelFunction(self, pixel):
        for y in range(0,self.rows):
            rowFunction = [[], [], []] #RGB
            for x in range(0, self.cols):
                img = numpy.array(self.lfReader.openImage(y,x))
                pxVal = img[pixel[1]][pixel[0]]
                for c in range(0, self.channels):
                    rowFunction[c].append(pxVal[c])
            for c in range(0, self.channels):
                self.pxFunction[c].append(rowFunction[c])

    def makeplots2d(self):
        fig, axis = plt.subplots(3)
        for c in range(0, self.channels):
            axis[c].plot(self.pxFunction[c][0])
        plt.show()

    def makePlots3d(self):
        fig, axis = plt.subplots(3,1,subplot_kw=dict(projection='3d'))
        for c in range(0, self.channels):
            xVals = range(0, self.cols)
            yVals = range(0, self.rows)
            xVals, yVals = numpy.meshgrid(xVals, yVals)
            zVals = []
            axis[c].plot_surface(xVals, yVals, numpy.array(self.pxFunction[c]))
        plt.show()

    def makePlots(self):
        if self.rows == 1:
            self.makeplots2d()
        else:
            self.makePlots3d()

    def analyze(self):
        pixel = (int(sys.argv[2]), int(sys.argv[3]))
        self.getPixelFunction(pixel)
        self.makePlots()

try:
    pa = PixelAnalyzer()
    pa.loadInput()
    pa.analyze()
except Exception as e:
    print(e)
    print(traceback.format_exc())
