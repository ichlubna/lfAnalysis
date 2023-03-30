#If the files in the folder are named as row_column (starting from 0) the data are interpreted as 2D light field, otherwise as 1D

import os
from pathlib import Path
from PIL import Image

class LfReader:
    cols = 0
    rows = 0
    files = []
    path = ""
    pixFmt = "RGB"

    def loadDir(self, path):
        self.path = path
        files = sorted(os.listdir(path))
        length = Path(files[-1]).stem.split("_")
        if len(length) == 1:
            self.cols = len(files)
            self.rows = 1
        else:
            self.cols = int(length[1])+1
            self.rows = int(length[0])+1
        self.files = [files[i:i+self.cols] for i in range(0, len(files), self.cols)]

    def getColsRows(self):
        return [self.cols, self.rows]

    def getResolution(self):
        return self.openImage(0,0).size

    def setPixFmt(self, pixFmt):
        self.pixFmt = pixFmt

    def openImage(self, row, column):
        filePath = os.path.join(self.path, self.files[row][column])
        image = Image.open(filePath)
        return image.convert(self.pixFmt)

    def getRowFunction(self, coordY, imgX, imgY):
        image = self.openImage(imgY, imgX)
        pixels = [ [0] for i in range(0, image.size[0])]
        for x in range(0, image.size[0]):
            pixels[x] = image.getpixel((x, coordY))
        return pixels

    def getPxFunction(self, coordX, coordY):
        pixel = [ [0]*self.cols for i in range(0, self.rows)]
        for x in range(0, self.cols):
            for y in range(0, self.rows):
                image = self.openImage(y, x)
                pixel[y][x] = image.getpixel((coordX, coordY))
        return pixel
