from task1_naive_python import benchmark_naive
from task2_numpy import benchmark_numpy
from task3_cupy_fp32 import benchmark_cupy_fp32
from task3_cupy_fp64 import benchmark_cupy_fp64
from plot_results import plot_results
import subprocess

sizes = [64, 128] 
iterations = 30

# Task 1
results_naive = benchmark_naive(sizes, iterations)
plot_results(results_naive, "Python Loop Performance", "graph1_python_loop.png")

# Task 2
results_numpy = benchmark_numpy(sizes, iterations)
plot_results(results_numpy, "NumPy Performance", "graph2_numpy.png")

# Task 3.1
for size in sizes:
    results_cupy_fp32 = benchmark_cupy_fp32([64, 128, 256, 512, 1024], iterations)
    print(f"Size {size} results:", results_cupy_fp32)
plot_results(results_cupy_fp32, "CuPy FP32 Performance", "graph3_cupy_fp32.png")

# Task 3.2
for size in sizes:
    results_cupy_fp64 = benchmark_cupy_fp64([64, 128, 256, 512, 1024], iterations)
    print(f"Size {size} results:", results_cupy_fp64)
plot_results(results_cupy_fp64, "CuPy FP64 Performance", "graph4_cupy_fp64.png")

# Task 4 (C code)
results_c = {}
for n in sizes:
    process = subprocess.Popen(["./matmul_c", str(n), str(iterations)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, _ = process.communicate()

    # Decode bytes to string and split cleanly into lines
    lines = stdout.decode().strip().split("\n")

    # Parse only valid lines with exactly 2 comma-separated values
    times = []
    for line in lines:
        parts = line.strip().split(",")
        if len(parts) == 2:
            try:
                times.append(float(parts[1]))
            except ValueError:
                pass  # skip bad lines

    gflops_list = [2 * (n ** 3) / (t * 1e9) for t in times]
    import numpy as np
    results_c[n] = (np.mean(gflops_list), np.std(gflops_list))

plot_results(results_c, "C Loop Performance", "graph5_c_loop.png")


