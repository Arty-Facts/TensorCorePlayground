#pragma once
#include <mma.h>
#include "cuda_fp16.h"

template<typename T>
__global__ void matmul_kernel(T* C, T* A, T* B, const unsigned int N) {

    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    T tmpSum = 0.0;
#pragma unroll
    for (unsigned int i = 0; i < N; i++) {
        // assuming the matrix is transposed for better Coalescing
        // tmpSum += A[y * N + i] * B[i * N + x];
        tmpSum += A[y * N + i] * B[x * N + i];
    }
    C[y * N + x] = tmpSum;

}

template<>
__global__ void matmul_kernel<half>(half* C, half* A, half* B, const unsigned int N) {

    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    half2 *A2 = reinterpret_cast<half2*>(A);
    half2 *B2 = reinterpret_cast<half2*>(B);

    half2 tmpSum(0.f, 0.f);
#pragma unroll
    for (unsigned int i = 0; i < N/2; i++) {
        // assuming the matrix is transposed for better Coalescing
        // tmpSum += A[y * N + i] * B[i * N + x];
        tmpSum += A2[y * N/2 + i] * B2[x * N/2 + i];
    }
    C[y * N + x] = tmpSum.x + tmpSum.y;

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
    unsigned int x, y, k, b, dx, dy;
    x = blockIdx.x * blockDim.x + threadIdx.x;
    y = blockIdx.y * blockDim.y + threadIdx.y;

    T sum = 0.0;

    for (b = 0; b < gridDim.x; b++)
    {
        __shared__ T As[BLOCK_SIZE * BLOCK_SIZE];
        __shared__ T Bs[BLOCK_SIZE * BLOCK_SIZE];
        // Index locked to patch
        dx = b * blockDim.x + threadIdx.x;
        dy = b * blockDim.y + threadIdx.y;
        As[threadIdx.y * blockDim.x + threadIdx.x] = A[N * y + dx];
        Bs[threadIdx.y * blockDim.x + threadIdx.x] = B[N * x + dy];
        __syncthreads(); // Synchronize to make sure all data is loaded
        // Loop, perform computations in patch
#pragma unroll
        for (k = 0; k < blockDim.x; ++k)
            sum += As[threadIdx.y * blockDim.x + k] * Bs[k * blockDim.x + threadIdx.x];
        __syncthreads(); // Synch so nobody starts next pass prematurely
    }
    C[x * N + y] = sum;
}

template <>
__global__ void matmul_opt_kernel<half>(half* C, half* A, half* B, const unsigned int N) {
   // Global index for thread
    unsigned int x, y, k, b,dy, start, end;
    x = blockIdx.x * blockDim.x + threadIdx.x;
    y = blockIdx.y * blockDim.y + threadIdx.y;
    half2 *A2 = reinterpret_cast<half2*>(A);
    half2 *B2 = reinterpret_cast<half2*>(B);
    half2 tmp(0.f, 0.f);
    //half sum = 0.0f; 
    for (b = 0; b < gridDim.x; b++)
    {
        __shared__ half2 As[BLOCK_SIZE * BLOCK_SIZE/2+1];
        __shared__ half2 Bs[BLOCK_SIZE * BLOCK_SIZE/2+1];
        // Index locked to patch
        dy = b * blockDim.x/2 + threadIdx.y;
        if ( threadIdx.y < blockDim.y/2){
            As[threadIdx.y * blockDim.x + threadIdx.x] = A2[(N/2) * x + dy];
            Bs[threadIdx.y * blockDim.x + threadIdx.x] = B2[(N/2) * y + dy];
            // printf("%f, %f AS \n",  (float)A2[(N/2) * y + dx].x, 
            //                         (float)A2[(N/2) * y + dx].y);
        }else{
        }
        
        __syncthreads(); // Synchronize to make sure all data is loaded
        // Loop, perform computations in patch

        start = 0;
        end = blockDim.x/2;
#pragma unroll
        for (k = start; k < end; ++k){
            tmp += As[k * blockDim.x + threadIdx.y] * Bs[k * blockDim.y + threadIdx.x];
        }
        __syncthreads(); // Synch so nobody starts next pass prematurely
    }
    // printf("%f, %f Sum \n",  (float)sum.x, 
    //                          (float)sum.y);
    C[x * N + y] = tmp.x + tmp.y;
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

    nvcuda::wmma::fill_fragment(acc_frag, 0.0f);

    // Loop over k
#pragma unroll
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
#define CHUNK_LINE_BYTES (CHUNK * WMMA_C * sizeof(half))
#define WARP_COPY_BYTES (WARP_SIZE * sizeof(int4))
#define CHUNK_COPY_LINES_PER_WARP (WARP_COPY_BYTES / CHUNK_LINE_BYTES)
#define CHUNK_COPY_LINE_LANES (WARP_SIZE / CHUNK_COPY_LINES_PER_WARP)

#define WARPS_PER_BLOCK 8
#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4

#define WARP_ROW_TILES 4
#define WARP_COL_TILES 2
#define TILES 256

#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS)

#define GLOBAL_MEM_STRIDE MAT_SIZE

#define SHMEM_STRIDE (WMMA_C * BLOCK_ROW_TILES)
#define SHMEM_OFFSET (WMMA_C * WARP_ROW_TILES)


template<typename T, const int WARPSIZE>
__global__ void matmul_shared_mma_kernel(T* C, T* A, T* B, const unsigned int N) 
    {
        __shared__ half shmem[BLOCK_SIZE][CHUNK * WMMA_C + SKEW_HALF];

        // Warp and lane identification.
        const unsigned int warpId = threadIdx.x / WARPSIZE;
        const unsigned int laneId = threadIdx.x % WARPSIZE;

        // Offset in shared memory from which the B matrix is stored.
        const size_t shmem_idx_b_off = BLOCK_COL_TILES * WMMA_C;

        // This pointer is used to access the C and D matrix tiles this warp computes.
        T* shmem_warp_tile_ptr = (T*)&shmem[0][0] + (warpId / 2) * SHMEM_STRIDE * WMMA_C * 2 + (warpId % 2) * SHMEM_OFFSET;

        // This pointer is used to stream the C and D matrices block-wide tile to and from shared memory.
        T* shmem_warp_stream_ptr = (T*)&shmem[0][0] + warpId * SHMEM_STRIDE * WMMA_C;


        // Each CTA slides along the 128 x 128 tiles from the top left corner of the matrix to the
        // right and down, and selects the next tile to compute. Once there's no such tile,
        // all warps in this CTA exit.
        for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
            const unsigned int block_tile_i = ((block_pos * BLOCK_ROW_TILES) / TILES) * (BLOCK_COL_TILES);
            const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % TILES;

            // Stop when there are no more D matrix tiles to compute in this CTA.
            if (block_tile_i >= TILES) {
                break;
            }

            // This warp's pointer to the C matrix data to copy memory from to shared memory.
            const size_t gmem_idx = (block_tile_i + warpId) * WMMA_C * GLOBAL_MEM_STRIDE + block_tile_j * WMMA_C;
            const T* src_gmem_warp_stream_ptr = &C[gmem_idx];

            // Stream multiple C tiles to shared memory.
#pragma unroll
            for (int i = 0; i < WMMA_C; i++) {
                typedef int4 copy_t;

                *((copy_t*)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) =
                    *((copy_t*)(src_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId);
            }

            __syncthreads();

            // These fragments will accumulate the result of A and B matrix fragment multiplications
            // along the MAT_SIZE dimension.
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_C, WMMA_C, WMMA_C, T> c[WARP_COL_TILES][WARP_ROW_TILES];

            // Load the C matrix tiles into fragments from shared memory.
#pragma unroll
            for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
                for (int j = 0; j < WARP_ROW_TILES; j++) {
                    const T* tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * WMMA_C + j * WMMA_C;

                    nvcuda::wmma::load_matrix_sync(c[i][j], tile_ptr, SHMEM_STRIDE, nvcuda::wmma::mem_row_major);
                }
            }

            __syncthreads();
            // Select what warp copies what matrix to shared memory.
            // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
            const half* warp_ptr = (warpId < 4) ? (&A[block_tile_i * WMMA_C * MAT_SIZE] + WMMA_C * MAT_SIZE * (warpId % 4) * 2) :
                (&B[block_tile_j * WMMA_C * MAT_SIZE] + WMMA_C * MAT_SIZE * (warpId % 4) * 2);

            // Go through the global K dimension by a fixed step at a time.
#pragma unroll
            for (int tile_k = 0; tile_k < TILES; tile_k += CHUNK) {
                // Copy slices of the A and B matrices to shared memory.
                // The first half of the warps in the CTA copy the A matrix, the rest copy the B matrix.
                size_t shmem_idx = warpId < (WARPS_PER_BLOCK / 2) ? (WMMA_C * (warpId % (WARPS_PER_BLOCK / 2)) * 2) :
                    (WMMA_C * (warpId % (WARPS_PER_BLOCK / 2)) * 2 + shmem_idx_b_off);

                // First half of the warp copies the first row / column of the matrix,
                // the second half of the warp copies the next.
                int4* lane_ptr = (int4*)(warp_ptr + tile_k * WMMA_C + (laneId / CHUNK_COPY_LINE_LANES) * MAT_SIZE) + (laneId % CHUNK_COPY_LINE_LANES);

                // Shift the second half of the warp to the next row / column in the shared memory.
                shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

#pragma unroll
                for (int i = 0; i < ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP) * 2; i++) {
                    // Copy 16 bytes at once in each lane.
                    *((int4*)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) = *lane_ptr;

                    // Advance the global memory pointer and the shared memory index.
                    lane_ptr = (int4*)((half*)lane_ptr + MAT_SIZE * CHUNK_COPY_LINES_PER_WARP);
                    shmem_idx += CHUNK_COPY_LINES_PER_WARP;
                }

                __syncthreads();

                // Compute a grid of C matrix tiles in each warp.
#pragma unroll
                for (int k_step = 0; k_step < CHUNK; k_step++) {
                    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_C, WMMA_C, WMMA_C, T, nvcuda::wmma::row_major> a[WARP_COL_TILES];
                    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_C, WMMA_C, WMMA_C, T, nvcuda::wmma::col_major> b[WARP_ROW_TILES];

#pragma unroll
                    for (int i = 0; i < WARP_COL_TILES; i++) {
                        size_t shmem_idx_a = (warpId / 2) * WMMA_C * 2 + (i * WMMA_C);
                        const T* tile_ptr = &shmem[shmem_idx_a][k_step * WMMA_C];

                        nvcuda::wmma::load_matrix_sync(a[i], tile_ptr, WMMA_C * CHUNK + SKEW_HALF);

#pragma unroll
                        for (int j = 0; j < WARP_ROW_TILES; j++) {
                            if (i == 0) {
                                // Load the B matrix fragment once, because it is going to be reused
                                // against the other A matrix fragments.
                                size_t shmem_idx_b = shmem_idx_b_off + (WARP_ROW_TILES * WMMA_C) * (warpId % 2) + (j * WMMA_C);
                                const T* tile_ptr = &shmem[shmem_idx_b][k_step * WMMA_C];

                                nvcuda::wmma::load_matrix_sync(b[j], tile_ptr, WMMA_C * CHUNK + SKEW_HALF);
                            }

                            nvcuda::wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
                        }
                    }
                }

                __syncthreads();
            }

            // Store the D fragments to shared memory.
#pragma unroll
            for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
                for (int j = 0; j < WARP_ROW_TILES; j++) {
                    // Uniform, point-wise transformations of ALL fragment elements by ALL threads in the
                    // warp are well-defined even though element indices within fragment storage are not defined.

                    T* tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * WMMA_C + j * WMMA_C;

                    nvcuda::wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, nvcuda::wmma::mem_row_major);
                }
            }

            __syncthreads();

            // Now that shared memory contains all the D tiles, stream them to global memory.
            T* dst_gmem_warp_stream_ptr = &C[gmem_idx];

#pragma unroll
            for (int i = 0; i < WMMA_C; i++) {
                *((int4*)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
                    *((int4*)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
            }

            __syncthreads();
        }
    }