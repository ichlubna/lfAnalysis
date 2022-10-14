import lfreader as lfr
import evaluator as eva
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from progress.bar import ChargingBar
import numpy
import cv2
import sys
import os
import traceback

class KernelParams:
    name = ""
    outDir = ""
    module = None
    parameter = 0
    blockSize = (16,16,1)
    blockCount = (1,1)
    sharedSize = 0

    def __init__(self, name, outDir, module, parameter, blockSize, blockCount, sharedSize):
        self.name = name
        self.outDir = outDir
        self.module = module
        self.parameter = parameter
        self.blockSize = blockSize
        self.blockCount = blockCount
        self.sharedSize = sharedSize

class KernelTester:
    renderedViewsCount = 8
    imageCount = 64
    numberOfMeasurements = 10
    width = 0
    height = 0
    cols = 0
    rows = 0
    depth = 4
    size = 0
    warpSize = 32
    totalSize = 0
    lfReader = None
    imagesGPU = None
    resultGPU = None
    weightMatrixGPU = None
    kernels = []

    def loadInput(self):
        lfReader = lfr.LfReader()
        lfReader.setPixFmt("RGBA")
        path = sys.argv[1]
        lfReader.loadDir(path)
        colsRows = lfReader.getColsRows()
        self.cols = numpy.int32(colsRows[0])
        self.rows = numpy.int32(colsRows[1])
        self.imageCount = self.cols*self.rows
        width, height = lfReader.getResolution()
        self.width = numpy.int32(width)
        self.height = numpy.int32(height)
        depth = numpy.int32(4)
        self.size = int(width*height*depth)
        self.totalSize = numpy.uint(self.size*self.imageCount)
        self.lfReader = lfReader

    def computeWeights(self, coords):
        weights = numpy.zeros(self.cols*self.rows, numpy.half)
        maxDistance = numpy.linalg.norm(numpy.array((0,0)) - numpy.array(self.cols, self.rows))
        weightSum = 0
        for y in range(0, self.rows):
            for x in range(0, self.cols):
                weight = maxDistance - numpy.linalg.norm(numpy.array((x, y)) - numpy.array(coords))
                linear = self.cols*y + x
                weights[linear] = numpy.single(weight)
                weightSum += weight
        for i in range(0, weights.size):
            weights[i] /= weightSum
        return weights

    def createViewTrajectory(self, start, end):
        stepX = (end[0]-start[0])/self.renderedViewsCount
        stepY = (end[1]-start[1])/self.renderedViewsCount
        trajectory = []
        for i in range(0, self.renderedViewsCount):
            trajectory.append((start[0]+stepX*i, start[1]+stepY*i))
        return trajectory

    def createTrajectoryWeightMatrix(self, start, end):
        trajectory = self.createViewTrajectory(start, end)
        weightVectors = []
        for coords in trajectory:
            weights = self.computeWeights(coords)
            weightVectors.append(weights)
        weightMatrix= numpy.c_[ weightVectors ].T
        return weightMatrix

    def allocWeightMatrices(self):
        weightMatrix = self.createTrajectoryWeightMatrix((0,0), (self.cols, self.rows))
        byteWeights = weightMatrix.tobytes()
        self.weightMatrixGPU = cuda.mem_alloc(weightMatrix.size*2)
        cuda.memcpy_htod(self.weightMatrixGPU, byteWeights)

    def allocGPUResources(self):
        bar = ChargingBar("Allocating and uploading textures", max=self.cols*self.rows+2)
        self.allocWeightMatrices()
        bar.next()

        images = bytes()
        for y in range(0,self.rows):
            for x in range(0, self.cols):
                img = self.lfReader.openImage(x,y)
                imgBytes = img.tobytes()
                images += imgBytes
                bar.next()
        bar.finish()

        self.imagesGPU = cuda.mem_alloc(int(self.totalSize))
        cuda.memcpy_htod(self.imagesGPU, images)
        self.resultGPU = cuda.mem_alloc(self.size*self.renderedViewsCount)
        bar.next()
        bar.finish()
        print("")

    def compileKernels(self):
        kernelConstants = [ "-DIMG_WIDTH="+str(self.width), "-DIMG_HEIGHT="+str(self.height),
                            "-DGRID_COLS="+str(self.cols), "-DGRID_ROWS="+str(self.rows),
                            "-DWARP_SIZE="+str(self.warpSize), "-DWEIGHTS_COLS="+str(self.renderedViewsCount),
                            "-DWEIGHTS_ROWS="+str(self.cols*self.rows), "-DCHANNEL_COUNT="+str(self.depth)]
        scriptPath = os.path.dirname(os.path.realpath(__file__))
        kernelSourceMain =  open(scriptPath+"/cudaKernels/mainInterpolation.cu", "r").read()
        kernelSourceGeneral = open(scriptPath+"/cudaKernels/generalInterpolation.cu", "r").read()
        kernelSourcePerPixel = open(scriptPath+"/cudaKernels/perPixelInterpolation.cu", "r").read()
        kernelSourceTensorInter = open(scriptPath+"/cudaKernels/tensorInterpolation.cu", "r").read()

        perPixelKernel = SourceModule(kernelSourceGeneral+kernelSourcePerPixel+kernelSourceMain, options=kernelConstants, no_extern_c=True)
        tensorInterpolationKernel = SourceModule(kernelSourceGeneral+kernelSourceTensorInter+kernelSourceMain, options=kernelConstants, no_extern_c=True)

        self.kernels = [KernelParams("Per pixel", "classicInterpolation/", perPixelKernel, numpy.int32(1), (256,1,1), (int(round(self.width/(256))), int(self.height)), 1024),
                        KernelParams("Tensor", "tensorInterpolation/", tensorInterpolationKernel, numpy.int32(1), (self.warpSize*8,1,1), (int(self.width/64), int(self.height)), 8*8*4*(16)*2 + 64*8*2 + 8*8*4*8*2) ]


    def runKernels(self):
        #result = numpy.zeros((self.height, self.width, self.depth), numpy.uint8)
        result = numpy.zeros((self.size*self.renderedViewsCount, 1, 1), numpy.uint8)
        for kernel in self.kernels:
            print(kernel.name)
            for i in range(0,self.numberOfMeasurements):
                start=cuda.Event()
                end=cuda.Event()
                start.record()
                func = kernel.module.get_function("process")
                func(self.imagesGPU, self.resultGPU, self.weightMatrixGPU, kernel.parameter, block=kernel.blockSize, grid=kernel.blockCount, shared=kernel.sharedSize)
                end.record()
                end.synchronize()
                print("Time: "+str(start.time_till(end))+" ms")

                if i == self.numberOfMeasurements-1:
                    bar = ChargingBar("Downloading data and storing", max=self.renderedViewsCount+2)
                    cuda.memcpy_dtoh(result, self.resultGPU)
                    bar.next()
                    result = result.astype(numpy.uint8)
                    resultParts = numpy.split(result, self.renderedViewsCount)
                    bar.next()
                    for j in range(0, self.renderedViewsCount):
                        resultImage = Image.frombytes("RGBA", (self.width, self.height), resultParts[j])
                        path = "./"+kernel.outDir+str(j)+".png"
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        resultImage.save(path)
                        bar.next()
                    bar.finish()
                    print("")

try:
    kt = KernelTester()
    kt.loadInput()
    kt.compileKernels()
    kt.allocGPUResources()
    kt.runKernels()
    evaluator = eva.Evaluator()
    print("Comparing the results")
    print("Tensor interpolation:")
    print(evaluator.metrics("./classicInterpolation", "./tensorInterpolation"))
    print("")
except Exception as e:
    print(e)
    print(traceback.format_exc())
