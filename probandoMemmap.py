import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy import misc
from numpy import genfromtxt


def copia_vector_memmap(nImagesFile, inputFile, outputFile, nImages=60000):

    i = 0
    currentNImages = nImages
    contImages = 0
    fp = np.memmap(outputFile, dtype='float32', mode='w+', shape=(nImagesFile, 80))

    while currentNImages == nImages and i*nImages < nImagesFile:
        trainN = genfromtxt(inputFile, delimiter='\t', max_rows=nImages,
                            skip_header=nImages * i)
        X_train = trainN[:, 0:80]
        X_train = np.array(X_train)

        currentNImages = X_train.shape[0]
        contImages += currentNImages/60
        print(int(contImages))

        fp[i * nImages:i * nImages + currentNImages, :] = X_train[:, :]
        i += 1

    fp.flush()


def tamano_vector(inputFile):
    totalSize = 0
    i=0
    size = 60000
    while 60000 == size:
        sizeVector = genfromtxt(inputFile, delimiter='\t', max_rows=60000, skip_header=60000 * i)
        sizeVector = sizeVector[:, 0:80]
        sizeVector = np.array(sizeVector)
        size = sizeVector.shape[0]
        totalSize += size
        i += 1
    return totalSize


def genera_vector_memmap(nFpIn, testPercentage, inputFiles, outputFiles, nImages=60000):
    i = 0
    contImages = 0
    nfpTest = (nFpIn[0] + nFpIn[1])*testPercentage
    resto = nfpTest%60
    nfpTest = nfpTest-resto
    nfpTrain = (nFpIn[0] + nFpIn[1]) - nfpTest
    fpTrain = np.memmap(outputFiles[0], dtype='float32', mode='w+', shape=(int(nfpTrain), 80))
    fpTest = np.memmap(outputFiles[1], dtype='float32', mode='w+', shape=(int(nfpTest), 80))

    print("tamanos de nuevos vectores: " + str(nfpTrain) + " " + str(nfpTest))

    j = 0
    accImagesTest = 0
    accImagesTrain = 0
    for input in inputFiles:
        while contImages < nFpIn[j]:
            trainN = genfromtxt(input, delimiter='\t', max_rows=nImages,
                                skip_header=nImages * i)
            X_train = trainN[:, 0:80]
            X_train = np.array(X_train)

            contImages += X_train.shape[0]
            print(contImages/60)

            res = (X_train.shape[0] * (1 - testPercentage))%60
            imagesTrain = (X_train.shape[0] * (1 - testPercentage)) - res
            imagesTest = X_train.shape[0] - imagesTrain

            if (imagesTrain + accImagesTrain) > nfpTrain:
                dif = (imagesTrain + accImagesTrain) - nfpTrain
                imagesTrain -= dif
                imagesTest += dif
            if (imagesTest + accImagesTest) > nfpTest:
                dif = (imagesTest + accImagesTest) - nfpTest
                imagesTrain += dif
                imagesTest -= dif

            fpTrain[int(accImagesTrain):int(accImagesTrain + imagesTrain), :] = \
            X_train[0:int(imagesTrain), :]

            fpTest[int(accImagesTest):int(accImagesTest + imagesTest), :] = \
                X_train[int(X_train.shape[0] - imagesTest):X_train.shape[0], :]

            accImagesTest += imagesTest
            accImagesTrain += imagesTrain
            print("Imagenes guardadas en test: " + str(accImagesTest/60))
            print("Imagenes guardadas en train: " + str(accImagesTrain/60))

            i += 1
        j += 1
        i = 0
        contImages = 0

    fpTrain.flush()
    fpTest.flush()

'''
fp = np.memmap("negative_depth_memmap", dtype='float32', mode='r', shape=(nImagesFile, 80))

print(fp[nImagesFile-1, :])
print(np.shape(fp))

trainN = genfromtxt('/home/dugo/Bases de datos TFG/negative_depth', delimiter='\t', max_rows=nImages,
                    skip_header=nImagesFile - nImages)
X_train = trainN[:,0:80]
X_train = np.array(X_train)

print(X_train[nImages-1, :])
'''

nImagesFile = 60*31410
nImages = 60*1000
inputFile = '/home/dugo/Bases de datos TFG/negative_depth_09_06'
outputFile = "negative_depth_09_06_memmap"

fpIn = [20130*60, 13560*60]
per = 0.1
inputFiles = ["/home/dugo/Bases de datos TFG/positive_depth", "/home/dugo/Bases de datos TFG/positive_depth_09_21"]
outputFiles = ["positive_train", "positive_test"]
genera_vector_memmap(fpIn, per, inputFiles, outputFiles)
# copia_vector_memmap(nImagesFile, inputFile, outputFile)



