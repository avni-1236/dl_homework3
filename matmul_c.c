#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_seconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: %s <n> <iterations>\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);
    int iterations = atoi(argv[2]);
    size_t N = (size_t)n * n;

    double *A = malloc(N * sizeof(double));
    double *B = malloc(N * sizeof(double));
    double *C = malloc(N * sizeof(double));
    if (!A || !B || !C) { fprintf(stderr, "Allocation failed\\n"); return 2; }

    srand((unsigned)time(NULL));
    for (size_t i = 0; i < N; ++i) {
        A[i] = (double)rand() / RAND_MAX;
        B[i] = (double)rand() / RAND_MAX;
    }

    for (int it = 0; it < iterations; ++it) {
        for (size_t i = 0; i < N; ++i) C[i] = 0.0;
        double t0 = now_seconds();
        for (int i = 0; i < n; ++i) {
            for (int k = 0; k < n; ++k) {
                double aik = A[i*n + k];
                for (int j = 0; j < n; ++j) {
                    C[i*n + j] += aik * B[k*n + j];
                }
            }
        }
        double t1 = now_seconds();
        printf("%d,%.9f\n", it, t1 - t0);

    }

    free(A); free(B); free(C);
    return 0;
}


//gcc matmul_c.c -o matmul_c -lm
