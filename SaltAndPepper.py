from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from numba import cuda
import numba
import math
import os
import skimage
from time import time


TPB = 32


def addSaPNoise(image, amount):
    imageArray = np.array(image).copy()
    imageArrayNoisy = skimage.util.random_noise(imageArray, mode='s&p', amount=amount, clip=True)
    return imageArrayNoisy


def cpuCalc(a):
    b = a.copy()
    start = time()
    for i in range(2, a.shape[0] - 1):
        for j in range(2, a.shape[1] - 1):
            t = [
                a[i - 1][j - 1], a[i - 1][j], a[i - 1][j + 1],
                a[i][j - 1], a[i][j], a[i][j + 1],
                a[i + 1][j - 1], a[i + 1][j], a[i + 1][j + 1]
            ]
            t.sort()
            b[i][j] = t[(int)(len(t) / 2)]
    return b, time() - start


@cuda.jit
def gpuCalc(a, b):
    i, j = cuda.grid(2)
    t = cuda.local.array(shape=9, dtype=numba.int64)
    t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8] = a[i - 1][j - 1], a[i - 1][j], a[i - 1][j + 1], a[i][j - 1], \
                                                           a[i][j], a[i][j + 1], a[i + 1][j - 1], a[i + 1][j], a[i + 1][
                                                               j + 1]
    for k in range(8):
        for l in range(8 - k):
            if t[l] > t[l + 1]:
                t[l], t[l + 1] = t[l + 1], t[l]
    b[i][j] = t[(int)(len(t) / 2)]


def prepareGpuCalc(a):
    b = a.copy()
    blockSize = (TPB, TPB)

    blockspergrid_x = int(math.ceil(a.shape[0] / blockSize[1]))
    blockspergrid_y = int(math.ceil(a.shape[1] / blockSize[0]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    a_global = cuda.to_device(a)
    b_global = cuda.to_device(b)

    start = time()
    gpuCalc[blockspergrid, blockSize](a_global, b_global)
    return b_global.copy_to_host(), time() - start


if __name__ == '__main__':
    inputImage = Image.open(os.path.abspath('D:\Wallpapers\WotLK.jpg')).convert("L")

    plt.imshow(inputImage, cmap='gray')
    plt.show()

    imageNoisy = addSaPNoise(inputImage, 0.15)
    plt.imshow(imageNoisy, cmap='gray')
    plt.show()

    cpuNoiseImage, cpuTime = cpuCalc(np.array(inputImage))
    print('CPU Time: ', cpuTime)
    plt.imshow(cpuNoiseImage, cmap='gray')
    plt.show()

    gpuNoiseImage, gpuTime = prepareGpuCalc(np.array(inputImage))
    print('GPU Time: ', gpuTime)
    plt.imshow(gpuNoiseImage, cmap='gray')
    plt.show()
