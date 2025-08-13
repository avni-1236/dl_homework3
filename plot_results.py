import matplotlib.pyplot as plt

def plot_results(results, title, filename):
    sizes = list(results.keys())
    means = [results[n][0] for n in sizes]
    stds = [results[n][1] for n in sizes]

    plt.errorbar(sizes, means, yerr=stds, fmt='-o', capsize=5)
    plt.xlabel('Matrix Size (n)')
    plt.ylabel('Mean GFLOPS')
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
