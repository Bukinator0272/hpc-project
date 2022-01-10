import numpy as np
from numba import cuda, int32
from time import time
import matplotlib.pyplot as plt

TPB = 32


def initMatrix(n):
    size = (n, n)
    a = np.random.randint(256, size=size)
    b = np.random.randint(256, size=size)
    return a, b


def cpuMult(a, b):
    start = time()
    c = a.dot(b)
    return time() - start


@cuda.jit
def gpuMult(A, B, C):
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=int32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=int32)
    x, y = cuda.grid(2)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x

    tempCount = 0.
    for i in range(bpg):
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]
        cuda.syncthreads()
        for j in range(TPB):
            tempCount += sA[tx, j] * sB[j, ty]
        cuda.syncthreads()
    C[x, y] = tempCount


def prepareGpuMult(a, b, n):
    n_tpb = int(n / TPB)
    gridSize = (n_tpb, n_tpb)
    blockSize = (TPB, TPB)
    c = np.zeros((n, n))

    start = time()
    A = cuda.to_device(a)
    B = cuda.to_device(b)
    C = cuda.to_device(c)
    gpuMult[gridSize, blockSize](A, B, C)
    c = C.copy_to_host()
    return time() - start


def calc(size, count):
    gpuTimeSum = 0
    cpuTimeSum = 0
    for _ in range(count):
        a, b = initMatrix(size)
        cpuTimeSum += cpuMult(a, b)
        gpuTimeSum += prepareGpuMult(a, b, size)

    print('Matrix dimension:', size)
    cpuTime = cpuTimeSum / count
    print('CPU time:', cpuTime)
    gpuTime = gpuTimeSum / count
    print('GPU time :', gpuTime)
    return cpuTime / gpuTime


if __name__ == '__main__':
    attemptCounter = 3
    sizes = [64, 128, 256, 512, 1024, 2048]
    accelerationArr = [calc(tempSize, attemptCounter) for tempSize in sizes]

    plt.plot(np.array(sizes), np.array(accelerationArr))
    plt.xlabel('Matrix dimension (nxn)')
    plt.ylabel('Acceleration (CPU_time / GPU_time)')

    plt.show()
