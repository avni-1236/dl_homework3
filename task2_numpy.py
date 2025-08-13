import time
import numpy as np

def benchmark_numpy(sizes, iterations):
    results = {}
    for n in sizes:
        A = np.random.rand(n, n).astype(np.float64)
        B = np.random.rand(n, n).astype(np.float64)
        gflops_list = []
        for _ in range(iterations):
            start = time.perf_counter()
            C = A @ B
            end = time.perf_counter()
            flops = 2 * (n ** 3)
            gflops_list.append(flops / ((end - start) * 1e9))
        results[n] = (np.mean(gflops_list), np.std(gflops_list))
    return results
