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
        if (n > BLOCK_SIZE) {
            threadsPerBlock.x = BLOCK_SIZE;
            threadsPerBlock.y = BLOCK_SIZE;
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
        cudaStatus = cudaSetDevice(DEVICE);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
            return cudaStatus;
        }

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, DEVICE);
        printf("For device #%d\n", DEVICE);
        printf("Device name:                %s\n", deviceProp.name);
        printf("Major revision number:      %d\n", deviceProp.major);
        printf("Minor revision Number:      %d\n", deviceProp.minor);
        printf("Total Global Memory:        %d\n", deviceProp.totalGlobalMem);
        printf("Total shared mem per block: %d\n", deviceProp.sharedMemPerBlock);
        printf("Total const mem size:       %d\n", deviceProp.totalConstMem);
        printf("Warp size:                  %d\n", deviceProp.warpSize);
        printf("Maximum block dimensions:   %d x %d x %d\n", deviceProp.maxThreadsDim[0], \
            deviceProp.maxThreadsDim[1], \
            deviceProp.maxThreadsDim[2]);

        printf("Maximum grid dimensions:    %d x %d x %d\n", deviceProp.maxGridSize[0], \
            deviceProp.maxGridSize[1], \
            deviceProp.maxGridSize[2]);
        printf("Clock Rate:                 %d\n", deviceProp.clockRate);
        printf("Number of muliprocessors:   %d\n", deviceProp.multiProcessorCount);

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
