import lfreader as lfr
import evaluator as eva
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from progress.bar import ChargingBar
import numpy
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
    focus = 10
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
    textures = None
    resultGPU = None
    weightMatrixGPU = None
    rows_per_group = 16
    matrix_load_once = True
    weights_cols_major = True
    persistent_threads = True
    sync_every_row = True
    useTextures = False
 
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
        if self.weights_cols_major:
            weightsVectorsTransformed = []
            for block in range(int(self.rows * self.cols/16)):
                for weight in weightVectors:
                    weightsVectorsTransformed.append(weight[16*block:16*(block+1)])
            weightMatrix=numpy.c_[ weightsVectorsTransformed ]
        else:
            weightMatrix= numpy.c_[ weightVectors ].T
        return weightMatrix

    def allocWeightMatrices(self):
        weightMatrix = self.createTrajectoryWeightMatrix((0,0), (self.cols, self.rows))
        byteWeights = weightMatrix.tobytes()
        self.weightMatrixGPU = cuda.mem_alloc(weightMatrix.size*2)
        self.zeroAtomicCounterGPU = cuda.mem_alloc(4)
        self.atomicCounterGPU = cuda.mem_alloc(4)
        cuda.memcpy_htod(self.zeroAtomicCounterGPU, numpy.array([0],dtype=numpy.uint32))
        cuda.memcpy_htod(self.weightMatrixGPU, byteWeights) 

    def loadImagesAsTextures(self, images):
        descr = cuda.ArrayDescriptor3D()
        descr.width = int(self.width)
        descr.height = int(self.height)
        descr.depth = int(self.imageCount)
        descr.format = cuda.dtype_to_array_format(numpy.uint32)
        descr.num_channels = 1

        self.textures = cuda.Array(descr)
        copy = cuda.Memcpy3D()
        copy.set_src_host(images.view(">u4"))
        copy.set_dst_array(self.textures)
        copy.width_in_bytes = copy.src_pitch = images.strides[1]
        copy.src_height = copy.height = int(self.height)
        copy.depth = int(self.imageCount)
        copy()
        self.imagesGPU = cuda.mem_alloc(1)

    def loadImagesAsBuffers(self, images):
        self.imagesGPU = cuda.mem_alloc(int(self.totalSize))
        cuda.memcpy_htod(self.imagesGPU, images)

    def allocGPUResources(self):
        bar = ChargingBar("Allocating and uploading textures", max=self.cols*self.rows+2)
        self.allocWeightMatrices()
        bar.next()

        images = numpy.zeros((self.cols*self.rows, self.height, self.width, self.depth), numpy.uint8)
        for y in range(0,self.rows):
            for x in range(0, self.cols):
                img = numpy.array(self.lfReader.openImage(x,y))
                images[y*self.cols+x][:] = img
                bar.next()
        bar.finish()

        if self.useTextures:
            self.loadImagesAsTextures(images)
        else:
            self.loadImagesAsBuffers(images)

        self.imageStartsGPU = cuda.mem_alloc(int(self.renderedViewsCount)*8)
        self.resultGPU = cuda.mem_alloc(self.size*self.renderedViewsCount)
        bar.next()
        bar.finish()
        print("")

    def compileKernels(self):
        scriptPath = os.path.dirname(os.path.realpath(__file__))

        kernelConstants = [ "-I="+scriptPath+"/cudaKernels",
                            "-DIMG_WIDTH="+str(self.width), "-DIMG_HEIGHT="+str(self.height),
                            "-DGRID_COLS="+str(self.cols), "-DGRID_ROWS="+str(self.rows),
                            "-DWARP_SIZE="+str(self.warpSize), "-DCHANNEL_COUNT="+str(self.depth),
                            "-DROWS_PER_THREAD="+str(self.rows_per_group), "-DOUT_VIEWS_COUNT="+str(self.renderedViewsCount)]
        if self.weights_cols_major:
            kernelConstants += ["-DWEIGHTS_COLS="+str(16),"-DWEIGHTS_ROWS="+str(int(self.cols*self.rows*self.renderedViewsCount/16))]
        else:
            kernelConstants += ["-DWEIGHTS_COLS="+str(self.renderedViewsCount),"-DWEIGHTS_ROWS="+str(self.cols*self.rows)]
        if self.matrix_load_once:
            kernelConstants.append("-DMATRIX_LOAD_ONCE")
        if self.weights_cols_major:
            kernelConstants.append("-DWEIGHTS_COL_MAJOR")
        if self.persistent_threads:
            kernelConstants.append("-DPERSISTENT_THREADS")
        if self.sync_every_row:
            kernelConstants.append("-DSYNC_EVERY_ROW")
        if self.useTextures:
            kernelConstants.append("-DUSE_TEXTURES")

        kernelSourceMain =  open(scriptPath+"/cudaKernels/mainInterpolation.cu", "r").read()
        kernelSourceGeneral = open(scriptPath+"/cudaKernels/generalInterpolation.cu", "r").read()
        kernelSourcePerPixel = open(scriptPath+"/cudaKernels/perPixelInterpolation.cu", "r").read()
        kernelSourceTensorInter = open(scriptPath+"/cudaKernels/tensorInterpolation.cu", "r").read()

        perPixelKernel = SourceModule(kernelSourcePerPixel, options=kernelConstants, no_extern_c=True)
        tensorInterpolationKernel = SourceModule(kernelSourceTensorInter, options=kernelConstants, no_extern_c=True)
        self.kernels = [KernelParams("Per pixel", "classicInterpolation/", perPixelKernel, numpy.int32(self.focus), (256,1,1), (int(round(self.width/(256))), int(self.height/self.rows_per_group)), 1024),
                       KernelParams("Tensor", "tensorInterpolation/", tensorInterpolationKernel, numpy.int32(self.focus), (self.warpSize*8,1,1), (int(self.width/256), int(self.height/self.rows_per_group)), 8*8*4*(16)*2 + 64*8*2) ]

    def runAndMeasureKernel(self, kernel):
        if self.useTextures:
            textureRef = kernel.module.get_texref("textures")
            textureRef.set_array(self.textures)
        if self.persistent_threads:
            cuda.memcpy_dtod(self.atomicCounterGPU, self.zeroAtomicCounterGPU,4)
        start=cuda.Event()
        end=cuda.Event()
        start.record()
        func = kernel.module.get_function("precalc_image_starts")
        func(self.imageStartsGPU, kernel.parameter, block=(int(self.cols * self.rows),1,1), grid=(1,1,1))
        func = kernel.module.get_function("process")
        func(self.imagesGPU, self.resultGPU, self.weightMatrixGPU, self.imageStartsGPU, self.atomicCounterGPU, block=kernel.blockSize, grid=kernel.blockCount, shared=kernel.sharedSize)
        end.record()
        end.synchronize()
        return start.time_till(end)

    def downloadAndSaveResult(self, kernel):
        result = numpy.zeros((self.size*self.renderedViewsCount, 1, 1), numpy.uint8)
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

    def runKernels(self):
        for kernel in self.kernels:
            print(kernel.name + " times [ms]:")
            for i in range(0,self.numberOfMeasurements):
                time = self.runAndMeasureKernel(kernel)
                print(str(time))
                if i == self.numberOfMeasurements-1:
                    self.downloadAndSaveResult(kernel)
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
