
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <chrono>
#include <cooperative_groups.h>
#include "cuda_fp16.h"
#include <mma.h>
#include <cuda.h>


#define BLOCK_SIZE 32
#define MAT_SIZE 1024*8
#define DEVICE 0
#define USE_CPU false
#define TYPE half
#define TC true
#define WARP_SIZE 32
#define VALIDATE true
#define WMMA_C 16

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
    
    T* a = reinterpret_cast<T*>(malloc(mem_size));
    T* b = reinterpret_cast<T*>(malloc(mem_size));
    T* c = reinterpret_cast<T*>(malloc(mem_size));

    float* verify_A = reinterpret_cast<float*>(malloc(sizeof(float) * SIZE));
    float* verify_B = reinterpret_cast<float*>(malloc(sizeof(float) * SIZE));
    float* verify_C =  reinterpret_cast<float*>(malloc(sizeof(float) * SIZE));

    // Initialize matrices on the host
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i * N + j] = (T)sin(i);
            b[i * N + j] = (T)cos(j);
            verify_A[i * N + j] = (float)sin(i);
            verify_B[i * N + j] = (float)cos(j);
        }
    }
    if (!use_cpu && VALIDATE) {
        matrixMultiplication<float> runne_float(verify_C, verify_A, verify_B, N);
        runne_float.setup(false);
        runne_float.lanch(&matmul_opt_kernel<float>);
        runne_float.therdown();
    }
    matrixMultiplication<T> runner(c, a, b, N);
    runner.setup();
    for (int i{}; i < 1; i++) {

  
        printf("Start...\n");
        printf("Computing %d x %d matrix ...\n", N ,N);
        printf("Type size %d\n", sizeof(T) );
        if (use_cpu) {
            cpu_time = matrixMultiplicationCPU<float>(verify_C, verify_A, verify_B, N);
            printf("matrixMultiplicationCPU CPU Computation time: %f\n", cpu_time);
        }

        gpu_time = runner.lanch(&matmul_kernel<T>);

        printf("matmul_kernel GPU Computation time: %f\n", gpu_time);

        runner.therdown();
        if (VALIDATE) {
            // Check the result and make sure it is correct
            err = validate(c, verify_C, N);
            printf("Error:  %lf\n", err);
        }

        gpu_time =  runner.lanch(&matmul_shared_kernel<T>);

        printf("matmul_shared_kernel GPU Computation time: %f\n", gpu_time);
        runner.therdown();
        if (VALIDATE) {
            // Check the result and make sure it is correct
            err = validate(c, verify_C, N);
            printf("Error:  %lf\n", err);
        }
        gpu_time = runner.lanch(&matmul_cuda_kernel<T>);

        printf("matmul_cuda_kernel GPU Computation time: %f\n", gpu_time);
        runner.therdown();
        if (VALIDATE) {
            // Check the result and make sure it is correct
            err = validate(c, verify_C, N);
            printf("Error:  %lf\n", err);
        }

        gpu_time = runner.lanch(&matmul_opt_kernel<T>);

        printf("matmul_opt_kernel GPU Computation time: %f\n", gpu_time);
        runner.therdown();
        if (VALIDATE) {
            // Check the result and make sure it is correct
            err = validate(c, verify_C, N);
            printf("Error:  %lf\n", err);
        }
        if (TC) {
            gpu_time = runner.lanch(&matmul_mma_kernel<T, WARP_SIZE>, true);

            printf("matmul_mma_kernel GPU Computation time: %f\n", gpu_time);
            runner.therdown();
            if (VALIDATE) {
                // Check the result and make sure it is correct
                err = validate(c, verify_C, N);
                printf("Error:  %lf\n", err);
            }
        }
    }
    free(verify_A);
    free(verify_B);
    free(verify_C);
    free(a);
    free(b);
    free(c);

}


int main()
{
    // un experiemt with the set up type
    run<TYPE>(USE_CPU);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}


