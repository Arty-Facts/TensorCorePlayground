
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <chrono>


#define BLOCKSIZE 32
template<typename T>
__global__ void matrixMultiplicationKernel(T* C, T* A, T* B,  unsigned int N) {

    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    T tmpSum = 0;

    if (y < N && x < N) {
        // each thread computes one element of the block sub-matrix
        for (unsigned int i = 0; i < N; i++) {
            tmpSum += A[y * N + i] * B[i * N + x];
        }
    }
    //printf("tmp sum: %f\n", tmpSum);
    C[y * N + x] = tmpSum;
    
}
template<typename T>
__global__ void matmul_kernel(T* C, T* A, T* B, unsigned int N) {

    __shared__ T sA[BLOCKSIZE][BLOCKSIZE];
    __shared__ T sB[BLOCKSIZE][BLOCKSIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    T tmpSum = 0;
    int k, kb;

    for (k = 0; k < N; k += blockDim.x) {
        __syncthreads();
        sA[ty][tx] = A[y * N + k + tx];
        sB[ty][tx] = B[(k + ty) * N + x];
        __syncthreads();

        for (kb = 0; kb < blockDim.x; kb++) {
            tmpSum += sA[ty][kb] * sB[kb][tx];
        }

    }

    C[y * N + x] = tmpSum;
}

template<typename T>
class matrixMultiplication
{
public:
    matrixMultiplication() = default;
    matrixMultiplication(T* c, const T* a, const T* b,  unsigned int n)
        : c{ c }, a{ a }, b{ b }, size{ n*n }, n{ n } {};

    float lanch(void (*kernal)(T* C, T* A, T* B, unsigned int N) )
    {

        cudaEvent_t myEventStart;
        cudaEvent_t myEventStop;
        float gpu_time;

        dim3 threadsPerBlock(n, n);
        dim3 blocksPerGrid(1, 1);
        if (n > BLOCKSIZE) {
            threadsPerBlock.x = BLOCKSIZE;
            threadsPerBlock.y = BLOCKSIZE;
            blocksPerGrid.x = ceil(double(n) / double(threadsPerBlock.x));
            blocksPerGrid.y = ceil(double(n) / double(threadsPerBlock.y));
        }
        // Time the run
        cudaEventCreate(&myEventStart);
        cudaEventRecord(myEventStart, 0);
        cudaEventSynchronize(myEventStart);
        (*kernal) <<<blocksPerGrid, threadsPerBlock >>> (dev_c, dev_a, dev_b, n);
        // wait to be done
        cudaThreadSynchronize();
        cudaEventCreate(&myEventStop);
        cudaEventRecord(myEventStop, 0);
        cudaEventSynchronize(myEventStop);
        cudaEventElapsedTime(&gpu_time, myEventStart, myEventStop);
        return gpu_time;
    }

    cudaError_t setup() 
    {
        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
            return cudaStatus;
        }

        // Allocate GPU buffers for three vectors (two input, one output)    .
        cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(T));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            return cudaStatus;
        }

        cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(T));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            return cudaStatus;
        }

        cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(T));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            return cudaStatus;
        }

        // Copy input vectors from host memory to GPU buffers.
        cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(T), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            return cudaStatus;
        }

        cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(T), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            return cudaStatus;
        }
        return cudaStatus;
    }
    cudaError_t therdown()
    {
        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "kernal launch failed: %s\n", cudaGetErrorString(cudaStatus));
            return cudaStatus;
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernal!\n", cudaStatus);
            return cudaStatus;
        }

        // Copy output vector from GPU buffer to host memory.
        cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(T), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            return cudaStatus;
        }
        return cudaStatus;
    }
    ~matrixMultiplication()
    {
        cudaFree(dev_c);
        cudaFree(dev_a);
        cudaFree(dev_b);
    }

private:
    T* c;
    const T* a;
    const T* b;
    unsigned int size;
    unsigned int n;
    T* dev_a = 0;
    T* dev_b = 0;
    T* dev_c = 0;
    cudaError_t cudaStatus;
};

template<typename T>
float matrixMultiplicationCPU(T* c, const T* a, const T* b, unsigned int N) {
    T sum;
    auto start_cpu_time = std::chrono::high_resolution_clock::now();
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            sum = 0.f;
            for (int n = 0; n < N; n++) {
                sum += a[row * N + n] * b[n * N + col];
            }
            c[row * N + col] = sum;
        }
    }
    auto end_cpu_time = std::chrono::high_resolution_clock::now();
    float cpu_time = std::chrono::duration<double, std::milli>(end_cpu_time - start_cpu_time).count();
    return cpu_time;

}
template<typename T>
double validate(T* val_c, T* c, unsigned int N) {
    double err = 0;
    // Check the result and make sure it is correct
    for (int ROW = 0; ROW < N; ROW++) {
        for (int COL = 0; COL < N; COL++) {
            //printf("cpu: %f -> gpu: %f\n", val_c[ROW * N + COL], c[ROW * N + COL]);
            err += val_c[ROW * N + COL] - c[ROW * N + COL];
        }
    }
    return err;
}

int main()
{
    // Perform matrix multiplication C = A*B
    // where A, B and C are NxN matrices
    const int N = 64;
    const int SIZE = N * N;

    // Allocate memory on the host
    float gpu_time;
    float cpu_time;
    double err;
    float a[SIZE];
    float b[SIZE];
    float c[SIZE];

    printf("Start...\n");
    printf("Computing %d x %d matrix ...\n", N ,N);
    // Initialize matrices on the host
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i * N + j] = sin(i);
            b[i * N + j] = cos(j);
        }
    }
    float* cpu_C;
    cpu_C = new float[SIZE];

    
    cpu_time = matrixMultiplicationCPU<float>(cpu_C, a, b, N);
    
    printf("CPU Computation time: %f\n", cpu_time);

    matrixMultiplication<float> runner(c, a, b, N);
    runner.setup();

    gpu_time = runner.lanch(&matrixMultiplicationKernel<float>);

    printf("matrixMultiplicationKernel GPU Computation time: %f\n", gpu_time);

    runner.therdown();

    // Check the result and make sure it is correct
    err = validate(c, cpu_C, N);
    printf("Error:  %lf\n", err);

    gpu_time =  runner.lanch(&matmul_kernel<float>);

    printf("matmul_kernel GPU Computation time: %f\n", gpu_time);

    runner.therdown();
    // Check the result and make sure it is correct
    err = validate(c, cpu_C, N);
    printf("Error:  %lf\n", err);
    delete cpu_C;

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}


