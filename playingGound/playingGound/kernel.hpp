
template<typename T>
__global__ void matmul_kernel(T* C, T* A, T* B, unsigned int N) {

    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    T tmpSum = 0;

    for (unsigned int i = 0; i < N; i++) {
        // assuming the matrix is transposed for better Coalescing
        // tmpSum += A[y * N + i] * B[i * N + x];
        tmpSum += A[y * N + i] * B[x * N + i];
    }
    C[y * N + x] = tmpSum;

}

template<typename T>
__global__ void matmul_shared_kernel(T* C, T* A, T* B, unsigned int N) {

    __shared__ T sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T sB[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    T tmpSum = 0;
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
__global__ void matmul_cuda_kernel(T* C, T* A, T* B, unsigned int N) {
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
    T Csub = 0;

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
__global__ void matmul_opt_kernel(T* C, T* A, T* B, unsigned int N) {
    // Global index for thread
    unsigned int i, j, k, b, ii, jj;
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;

    T sum = 0;

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
