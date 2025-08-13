import cupy as cp
import numpy as np
import time

def benchmark_cupy_fp32(n, iterations):
    # Generate random matrices on CPU using NumPy
    A_cpu = np.random.rand(n, n).astype(np.float32)
    B_cpu = np.random.rand(n, n).astype(np.float32)

    # Transfer them to GPU
    A = cp.asarray(A_cpu)
    B = cp.asarray(B_cpu)

    times = []
    for _ in range(iterations):
        cp.cuda.Stream.null.synchronize()  # Sync before timing
        start = time.time()
        C = cp.dot(A, B)
        cp.cuda.Stream.null.synchronize()  # Ensure GPU work is done
        end = time.time()
        times.append(end - start)

    return times
