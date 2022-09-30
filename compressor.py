import sys
#import rle
from scipy.signal import savgol_filter
from scipy.fftpack import dct, idct
from matplotlib import pyplot as plt

class Compressor:
    def analyze1D(self, values):
        coefs = dct(values, norm="ortho")
        dec = 10
        clamp = 128
        #for i in range(0,len(coefs)):
        #    coefs[i] = max(min(round(coefs[i]*dec), clamp),-clamp)/dec

        print(coefs)

        newCoefs = [0]*len(coefs)
        #sortedIndices = sorted(range(len(coefs)), key=lambda i: abs(coefs[i]))
        #for i in range(0,25):
        #    newCoefs[sortedIndices[i]] = coefs[sortedIndices[i]]
        #newCoefs[0] = coefs[0]

        for i in range(0, 25):
           newCoefs[i] = coefs[i]

        valuesBack = idct(newCoefs, norm="ortho")
        delta = []
        for i in range(len(coefs)):
            delta.append(round(abs(values[i]-valuesBack[i]),5))

        fig, axis = plt.subplots(3)
        axis[0].set_title("Input");
        axis[0].plot(values)
        axis[1].set_title("Output");
        axis[1].plot(valuesBack)
        axis[2].set_title("Delta");
        axis[2].plot(delta)
        plt.show()

        #print(*coefs)
        print("Size of data:")
        print(sys.getsizeof(values))
        print("Size of coefficients:")
        print(sys.getsizeof(coefs))
        #rleCoefs = rle.encode(coefs)
        #print("Size of rle")
        #print(sys.getsizeof(d))
        #print(sys.getsizeof(rle.decode(d[0], d[1])))
