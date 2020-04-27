
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
#define MAT_SIZE 1024
#define DEVICE 0
#define USE_CPU true
#define TYPE float
#define TC false
#define WARP_SIZE 32
#define VALIDATE true

#define WMMA_C 16
#define CHUNK 4
#define SKEW_HALF 8

#include "kernel.hpp"
#include "setup.hpp"
#include "utils.hpp"

void disp_flops(float time){
    printf("TFLOPS: %.2f\n", (((double)MAT_SIZE * MAT_SIZE * MAT_SIZE*2) / (time / 1000.)) / 1e12);
}

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
    
    T* a = new T[SIZE];
    T* b = new T[SIZE];
    T* c = new T[SIZE];

    float* verify_A = new float[SIZE];
    float* verify_B = new float[SIZE];
    float* verify_C = new float[SIZE];

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
        matrixMultiplication<float> runner_float(verify_C, verify_A, verify_B, N);
        runner_float.setup(false);
        runner_float.lanch(&matmul_opt_kernel<float>);
        runner_float.therdown();
    }
    matrixMultiplication<T> runner(c, a, b, N);
    runner.setup();
    for (int i{}; i < 1; i++) {
        
        printf("Start...\n");
        printf("Computing %d x %d matrix ...\n", N ,N);
        printf("Type size %d\n", (int)sizeof(T) );
        if (use_cpu) {
            cpu_time = matrixMultiplicationCPU<float>(verify_C, verify_A, verify_B, N);
            printf("matrixMultiplicationCPU CPU Computation time: %f\n", cpu_time);
            disp_flops(cpu_time);
        }
        
        gpu_time = runner.lanch(&matmul_kernel<T>);
        
        printf("matmul_kernel GPU Computation time: %f\n", gpu_time);
        disp_flops(gpu_time);
        
        runner.therdown();
        if (VALIDATE) {
            // Check the result and make sure it is correct
            err = validate(c, verify_C, N);
            printf("Error:  %lf\n", err);
        }
        
        gpu_time =  runner.lanch(&matmul_shared_kernel<T>);
        
        printf("matmul_shared_kernel GPU Computation time: %f\n", gpu_time);
        disp_flops(gpu_time);
        runner.therdown();
        if (VALIDATE) {
            // Check the result and make sure it is correct
            err = validate(c, verify_C, N);
            printf("Error:  %lf\n", err);
        }
        gpu_time = runner.lanch(&matmul_cuda_kernel<T>);
        
        printf("matmul_cuda_kernel GPU Computation time: %f\n", gpu_time);
        disp_flops(gpu_time);
        runner.therdown();
        if (VALIDATE) {
            // Check the result and make sure it is correct
            err = validate(c, verify_C, N);
            printf("Error:  %lf\n", err);
        }
        
        gpu_time = runner.lanch(&matmul_opt_kernel<T>);
        
        printf("matmul_opt_kernel GPU Computation time: %f\n", gpu_time);
        disp_flops(gpu_time);
        runner.therdown();
        if (VALIDATE) {
            // Check the result and make sure it is correct
            err = validate(c, verify_C, N);
            printf("Error:  %lf\n", err);
        }
        // dont compile this code if tensor cores not needed
        // float not supported on tensor cores
        if constexpr (TC) {
            gpu_time = runner.lanch(&matmul_mma_kernel<T, WARP_SIZE>, true);
            
            printf("matmul_mma_kernel GPU Computation time: %f\n", gpu_time);
            disp_flops(gpu_time);
            runner.therdown();
            if (VALIDATE) {
                // Check the result and make sure it is correct
                err = validate(c, verify_C, N);
                printf("Error:  %lf\n", err);
            }
            //gpu_time = runner.lanch(&matmul_shared_mma_kernel<T, WARP_SIZE>, true);

            //printf("matmul_shared_mma_kernel GPU Computation time: %f\n", gpu_time);
            //disp_flops(gpu_time);
            //runner.therdown();
            //if (VALIDATE) {
            //    // Check the result and make sure it is correct
            //    err = validate(c, verify_C, N);
            //    printf("Error:  %lf\n", err);
            //}

        }
    }
    delete[] verify_A;
    delete[] verify_B;
    delete[] verify_C;
    delete[] a;
    delete[] b;
    delete[] c;
    
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


