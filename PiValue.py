from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import numpy as np
import time
from tabulate import tabulate
import matplotlib.pyplot as plt

TPB = 16
ITER_COUNT = 16
BPG = int(ITER_COUNT / TPB)


@cuda.jit
def compute_pi(rng_states, N, res):
    thread_id = cuda.grid(1)
    z = 0
    for i in range(N):
        x = xoroshiro128p_uniform_float32(rng_states, thread_id)
        y = xoroshiro128p_uniform_float32(rng_states, thread_id)
        if x ** 2 + y ** 2 <= 1.0:
            z += 1

    res[thread_id] = 4.0 * z / N


def cpu_calculate_pi(iter_count, N):
    res = np.zeros(iter_count)
    for i in range(iter_count):
        x = np.random.uniform(size=N)
        y = np.random.uniform(size=N)
        z = x ** 2 + y ** 2 <= 1
        res[i] = 4.0 * sum(z) / N
    return res


def main(N):
    start = time.time()
    rng_states = create_xoroshiro128p_states(TPB * BPG, seed=1)
    out = np.zeros(TPB * BPG, dtype=np.float32)
    compute_pi[BPG, TPB](rng_states, N, out)
    gpu_pi = out.mean()
    gpu_time = time.time() - start

    start = time.time()
    cpu_pi = cpu_calculate_pi(ITER_COUNT, N).mean()
    cpu_time = time.time() - start

    return gpu_time, cpu_time, gpu_pi, cpu_pi, cpu_time / gpu_time


if __name__ == '__main__':
    rows = []
    for N in range(10_000, 100_000 + 1, 10_000):
        rows.append([N, *main(N)])
    print(tabulate(rows, headers=['N', 'gpu_time', 'cpu_time', 'gpu_pi', 'cpu_pi', 'acceleration']))

    plt.figure(figsize=(18, 4))

    plt.subplot(1, 3, 1)
    plt.title("cpu time")
    plt.plot(np.array(rows)[:, 0], np.array(rows)[:, 2])
    plt.xlabel("matrix size")
    plt.ylabel("time, ms")
    plt.grid()

    plt.subplot(1, 3, 2)
    plt.title("gpu time")
    plt.plot(np.array(rows)[:, 0], np.array(rows)[:, 1])
    plt.xlabel("matrix size")
    plt.ylabel("time, ms")
    plt.grid()

    plt.subplot(1, 3, 3)
    plt.title("acceleration")
    plt.plot(np.array(rows)[:, 0], np.array(rows)[:, 2] / np.array(rows)[:, 1])
    plt.xlabel("matrix size")
    plt.ylabel("cpu to gpu time ratio")
    plt.grid()

    plt.show()
