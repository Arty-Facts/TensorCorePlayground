#pragma once
#include <mma.h>
#include "cuda_fp16.h"

template<typename T>
__global__ void matmul_kernel(T* C, T* A, T* B, const unsigned int N) {

    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    T tmpSum = 0.0;

    for (unsigned int i = 0; i < N; i++) {
        // assuming the matrix is transposed for better Coalescing
        // tmpSum += A[y * N + i] * B[i * N + x];
        tmpSum += A[y * N + i] * B[x * N + i];
    }
    C[y * N + x] = tmpSum;

}

template<typename T>
__global__ void matmul_shared_kernel(T* C, T* A, T* B, const unsigned int N) {

    __shared__ T sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T sB[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    T tmpSum = 0.0;
    int k, kb;

    for (k = 0; k < N; k += blockDim.x) {
        __syncthreads();
        sA[ty][tx] = A[y * N + k + tx];
        sB[ty][tx] = B[x * N + k + ty];
        __syncthreads();

        for (kb = 0; kb < blockDim.x; kb++) {
            tmpSum += sA[ty][kb] * sB[kb][tx];
        }

    }

    C[y * N + x] = tmpSum;

}

template <typename T> 
__global__ void matmul_cuda_kernel(T* C, T* A, T* B, const unsigned int N) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = N * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd = aBegin + N - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = N * BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep = BLOCK_SIZE;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    T Csub = 0.0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
        a <= aEnd;
        a += aStep, b += bStep) {
        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ T As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ T Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + N * ty + tx];
        Bs[ty][tx] = B[b + N * tx + ty];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = N * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + N * ty + tx] = Csub;
}

template <typename T>
__global__ void matmul_opt_kernel(T* C, T* A, T* B, const unsigned int N) {
    // Global index for thread
    unsigned int i, j, k, b, ii, jj;
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;

    T sum = 0.0;

    for (b = 0; b < gridDim.x; b++)
    {
        __shared__ T As[BLOCK_SIZE * BLOCK_SIZE];
        __shared__ T Bs[BLOCK_SIZE * BLOCK_SIZE];
        // Index locked to patch
        ii = b * blockDim.x + threadIdx.x;
        jj = b * blockDim.y + threadIdx.y;
        As[threadIdx.y * blockDim.x + threadIdx.x] = A[N * j + ii];
        Bs[threadIdx.y * blockDim.x + threadIdx.x] = B[N * i + jj];
        __syncthreads(); // Synchronize to make sure all data is loaded
        // Loop, perform computations in patch
        for (k = 0; k < blockDim.x; ++k)
            sum += As[threadIdx.y * blockDim.x + k] * Bs[k * blockDim.x + threadIdx.x];
        __syncthreads(); // Synch so nobody starts next pass prematurely
    }
    C[i * N + j] = sum;
}

template <>
__global__ void matmul_opt_kernel<half>(half* C, half* A, half* B, const unsigned int N) {
   // Global index for thread
    unsigned int i, j, k, b, ii, start, end;
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    half2 *A2 = reinterpret_cast<half2*>(A);
    half2 *B2 = reinterpret_cast<half2*>(B);
    half2 sum(0.f, 0.f);
    for (b = 0; b < gridDim.x; b++)
    {
        __shared__ half2 As[BLOCK_SIZE * BLOCK_SIZE];
        __shared__ half2 Bs[BLOCK_SIZE * BLOCK_SIZE];
        // Index locked to patch
        int w  = (blockDim.x/2) + 1;
        int id = threadIdx.x+1;
        ii = b * (blockDim.x/2) + w%id;
        if (w < threadIdx.x){
            As[threadIdx.y * (blockDim.x/2) + w%id] = A2[N/2 * j + ii];
        }else{
            Bs[threadIdx.y * (blockDim.x/2) + w%id] = B2[N/2 * j + ii];
        }

        // printf("%f, %f h2 \n%f, %f h1 \n", (float)A2[N * j + ii].x, 
        //                                 (float)A2[N * j + ii].y, 
        //                                 (float)A[N * j + b * blockDim.x + threadIdx.x*2], 
        //                                 (float)A[N * j + b * blockDim.x + threadIdx.x*2+1]);
        __syncthreads(); // Synchronize to make sure all data is loaded
        // Loop, perform computations in patch

        start = 0;
        end = (blockDim.x/2);
        for (k = start; k < end; ++k){
            sum += As[threadIdx.y * (blockDim.x/2) + k] * Bs[threadIdx.y * (blockDim.x/2) + k];
        }
        __syncthreads(); // Synch so nobody starts next pass prematurely
    }
    C[i * N + j] = sum.x + sum.y;
}

template<   typename T, const int WARPSIZE >
__global__ void matmul_mma_kernel(T* C, T* A, T* B, const unsigned int N)
{

    // Tile using a 2D grid
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / WARPSIZE;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // Declare the fragments
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_C, WMMA_C, WMMA_C, T, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_C, WMMA_C, WMMA_C, T, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_C, WMMA_C, WMMA_C, T> acc_frag;

    nvcuda::wmma::fill_fragment(acc_frag, (T)0.0f);

    // Loop over k
    for (int i = 0; i < N; i += WMMA_C) {
        int aRow = warpM * WMMA_C;
        int bRow = warpN * WMMA_C;

        // Bounds checking
        if (aRow < N && i < N && bRow < N && i < N) {
            // Load the inputs
            nvcuda::wmma::load_matrix_sync(a_frag, A + i + aRow * N, N);
            nvcuda::wmma::load_matrix_sync(b_frag, B + i + bRow * N, N);

            // Perform the matrix multiplication
            nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // store
    int cCol = warpN * WMMA_C;
    int cRow = warpM * WMMA_C;

    if (cRow < N && cCol < N) {
        nvcuda::wmma::store_matrix_sync(C + cCol + cRow * N, acc_frag, N, nvcuda::wmma::mem_row_major);
    }
}

template<typename T, const int WARPSIZE>
__global__ void matmul_shared_mma_kernel(T* C, T* A, T* B, const unsigned int N) {

    
    // Tile using a 2D grid
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / WARPSIZE;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // Declare the fragments
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_C, WMMA_C, WMMA_C, T, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_C, WMMA_C, WMMA_C, T, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_C, WMMA_C, WMMA_C, T> acc_frag;

    nvcuda::wmma::fill_fragment(acc_frag, (T)0.0f);

    // Loop over k
    for (int i = 0; i < N; i += WMMA_C) {
        int aRow = warpM * WMMA_C;
        int bRow = warpN * WMMA_C;

        // Bounds checking
        if (aRow < N && i < N && bRow < N && i < N) {
            // Load the inputs
            nvcuda::wmma::load_matrix_sync(a_frag, A + i + aRow * N, N);
            nvcuda::wmma::load_matrix_sync(b_frag, B + i + bRow * N, N);

            // Perform the matrix multiplication
            nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // store
    int cCol = warpN * WMMA_C;
    int cRow = warpM * WMMA_C;

    if (cRow < N && cCol < N) {
        nvcuda::wmma::store_matrix_sync(C + cCol + cRow * N, acc_frag, N, nvcuda::wmma::mem_row_major);
    }

}