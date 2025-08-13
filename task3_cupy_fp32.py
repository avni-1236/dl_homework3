def benchmark_cupy_fp32(n, iterations):
    import cupy as cp
    import numpy as np
    import time

    # Handle list input
    if isinstance(n, (list, tuple)):
        results = {}
        for size in n:
            results[size] = benchmark_cupy_fp32(size, iterations)
        return results

    A_cpu = np.random.rand(n, n).astype(np.float32)
    B_cpu = np.random.rand(n, n).astype(np.float32)
    A = cp.asarray(A_cpu)
    B = cp.asarray(B_cpu)

    times = []
    for _ in range(iterations):
        cp.cuda.Stream.null.synchronize()
        start = time.time()
        C = cp.dot(A, B)
        cp.cuda.Stream.null.synchronize()
        end = time.time()
        times.append(end - start)

    return times
