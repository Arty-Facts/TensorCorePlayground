
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <chrono>
#include <cooperative_groups.h>
#include "cuda_fp16.h"


#define BLOCK_SIZE 32
#define MAT_SIZE 1024*4
#define DEVICE 0
#define USE_CPU false

#include "kernel.hpp"
#include "setup.hpp"
#include "utils.hpp"

template<typename T>
void run(bool use_cpu=true) {
    // Perform matrix multiplication C = A*B
    // where A, B and C are NxN matrices
    const int N = MAT_SIZE;
    const int SIZE = N * N;

    // Allocate memory on the host
    float gpu_time;
    double err;
    float cpu_time;
    unsigned int mem_size = sizeof(T) * SIZE;
    T* cpu_C;
    if (use_cpu) {
        cpu_C = reinterpret_cast<T*>(malloc(mem_size));
    }
    T* a = reinterpret_cast<T*>(malloc(mem_size));
    T* b = reinterpret_cast<T*>(malloc(mem_size));
    T* c = reinterpret_cast<T*>(malloc(mem_size));

    // Initialize matrices on the host
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i * N + j] = (T)sin(i);
            b[i * N + j] = (T)cos(j);
        }
    }

    matrixMultiplication<T> runner(c, a, b, N);
    runner.setup();
    for (int i{}; i < 1; i++) {

  
        printf("Start...\n");
        printf("Computing %d x %d matrix ...\n", N ,N);
        if (use_cpu) {
            cpu_time = matrixMultiplicationCPU<T>(cpu_C, a, b, N);
            printf("matrixMultiplicationCPU CPU Computation time: %f\n", cpu_time);
        }

        gpu_time = runner.lanch(&matmul_kernel<T>);

        printf("matmul_kernel GPU Computation time: %f\n", gpu_time);

        runner.therdown();
        if (use_cpu) {
            // Check the result and make sure it is correct
            err = validate(c, cpu_C, N);
            printf("Error:  %lf\n", err);
        }

        gpu_time =  runner.lanch(&matmul_shared_kernel<T>);

        printf("matmul_shared_kernel GPU Computation time: %f\n", gpu_time);
        runner.therdown();
        if (use_cpu) {
            // Check the result and make sure it is correct
            err = validate(c, cpu_C, N);
            printf("Error:  %lf\n", err);
        }
        gpu_time = runner.lanch(&matmul_cuda_kernel<T>);

        printf("matmul_cuda_kernel GPU Computation time: %f\n", gpu_time);
        runner.therdown();
        if (use_cpu) {
            // Check the result and make sure it is correct
            err = validate(c, cpu_C, N);
            printf("Error:  %lf\n", err);
        }

        gpu_time = runner.lanch(&matmul_opt_kernel<T>);

        printf("matmul_opt_kernel GPU Computation time: %f\n", gpu_time);
        runner.therdown();
        if (use_cpu) {
            // Check the result and make sure it is correct
            err = validate(c, cpu_C, N);
            printf("Error:  %lf\n", err);
        }
    }
    if (use_cpu)
        free(cpu_C);
    free(a);
    free(b);
    free(c);

}
//
//half2 operator=(const float& other ) {
//    return __float2half2_rn(other);
//}

int main()
{
    // un experiemt with the set up type
    run<float>(USE_CPU);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}


