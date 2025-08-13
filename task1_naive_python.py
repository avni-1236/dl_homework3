import time
import numpy as np

def matmul_naive(A, B, n):
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C

import time
import numpy as np

def matmul_naive(A, B, n):
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C

def benchmark_naive(sizes, iterations):
    results = {}
    for n in sizes:
        # ✅ Create random float matrices, not ranges
        A = [[float(np.random.rand()) for _ in range(n)] for _ in range(n)]
        B = [[float(np.random.rand()) for _ in range(n)] for _ in range(n)]
        gflops_list = []
        for _ in range(iterations):
            start = time.perf_counter()
            matmul_naive(A, B, n)
            end = time.perf_counter()
            flops = 2 * (n ** 3)
            gflops_list.append(flops / ((end - start) * 1e9))
        results[n] = (np.mean(gflops_list), np.std(gflops_list))
    return results
