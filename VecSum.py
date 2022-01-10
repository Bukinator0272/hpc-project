import numpy as np
from numba import cuda, int32
from time import time
import matplotlib.pyplot as plt

TPB = 256
cpuPlot = []
gpuPlot = []


def cpuSum(v):
    start = time()
    sum = np.sum(v)
    return time() - start


@cuda.jit
def gpuSum(v, sum):
    shared = cuda.shared.array(TPB, dtype=int32)
    thread = cuda.threadIdx.x
    block = cuda.blockIdx.x
    idx = block * TPB + thread
    shared[thread] = 0

    if idx < v.shape[0]:
        shared[thread] = v[idx]
        if thread == 0:
            S = 0
            for i in range(TPB):
                S += shared[i]
            cuda.atomic.add(sum, 0, S)


def prepareGpuSum(v, n):
    n_tpb = int(n / TPB)
    sum = np.zeros(1, dtype=np.int32)

    start = time()
    V = cuda.to_device(v)
    s = cuda.to_device(sum)
    gpuSum[n_tpb, TPB](V, s)
    sum = s.copy_to_host()
    return time() - start


def calc(length, count):
    gpuTimeSum = 0
    cpuTimeSum = 0
    for _ in range(count):
        v = np.random.randint(256, size=length)
        cpuTimeSum += cpuSum(v)
        gpuTimeSum += prepareGpuSum(v, length)

    print('Vector length:', length)

    cpuTime = cpuTimeSum / count
    print('CPU time:', cpuTime)
    cpuPlot.append(cpuTime)

    gpuTime = gpuTimeSum / count
    print('GPU time :', gpuTime)
    gpuPlot.append(gpuTime)

    return cpuTime / gpuTime


if __name__ == '__main__':
    attemptCounter = 3
    lengths = range(250000, 10000000, 250000)
    accelerations = [calc(tempLength, attemptCounter) for tempLength in lengths]

    plt.plot(np.array(lengths), np.array(accelerations))
    plt.xlabel('Vector length')
    plt.ylabel('Acceleration (CPU_time / GPU_time)')
    plt.show()

    plt.plot(lengths, cpuPlot, lengths, gpuPlot)
    plt.xlabel('Vector length')
    plt.ylabel('Calculation time')
    plt.legend(['CPU', 'GPU'])
    plt.show()
