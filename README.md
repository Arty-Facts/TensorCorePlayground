## block size 32
For device #0
Device name:                GeForce RTX 2080 SUPER
Major revision number:      7
Minor revision Number:      5
Total Global Memory:        0
Total shared mem per block: 49152
Total const mem size:       65536
Warp size:                  32
Maximum block dimensions:   1024 x 1024 x 64
Maximum grid dimensions:    2147483647 x 65535 x 65535
Clock Rate:                 1815000
Number of muliprocessors:   48
Start...
Computing 4096 x 4096 matrix ...
matmul_kernel GPU Computation time: 891.709961
matmul_shared_kernel GPU Computation time: 1171.492920
matmul_cuda_kernel GPU Computation time: 1161.099121
matmul_opt_kernel GPU Computation time: 903.343933

```
For device #1
Device name:                GeForce GTX 1060 3GB
Major revision number:      6
Minor revision Number:      1
Total Global Memory:        -1073741824
Total shared mem per block: 49152
Total const mem size:       65536
Warp size:                  32
Maximum block dimensions:   1024 x 1024 x 64
Maximum grid dimensions:    2147483647 x 65535 x 65535
Clock Rate:                 1708500
Number of muliprocessors:   9
Start...
Computing 4096 x 4096 matrix ...
matmul_kernel GPU Computation time: 8112.402344
matmul_shared_kernel GPU Computation time: 8558.845703
matmul_cuda_kernel GPU Computation time: 8571.676758
matmul_opt_kernel GPU Computation time: 10754.331055
```

## block size 16
```
For device #0
Device name:                GeForce RTX 2080 SUPER
Major revision number:      7
Minor revision Number:      5
Total Global Memory:        0
Total shared mem per block: 49152
Total const mem size:       65536
Warp size:                  32
Maximum block dimensions:   1024 x 1024 x 64
Maximum grid dimensions:    2147483647 x 65535 x 65535
Clock Rate:                 1815000
Number of muliprocessors:   48
Start...
Computing 4096 x 4096 matrix ...
matmul_kernel GPU Computation time: 880.493652
matmul_shared_kernel GPU Computation time: 1218.909058
matmul_cuda_kernel GPU Computation time: 1211.767944
matmul_opt_kernel GPU Computation time: 947.439880
```

```
For device #1
Device name:                GeForce GTX 1060 3GB
Major revision number:      6
Minor revision Number:      1
Total Global Memory:        -1073741824
Total shared mem per block: 49152
Total const mem size:       65536
Warp size:                  32
Maximum block dimensions:   1024 x 1024 x 64
Maximum grid dimensions:    2147483647 x 65535 x 65535
Clock Rate:                 1708500
Number of muliprocessors:   9
Start...
Computing 4096 x 4096 matrix ...
matmul_kernel GPU Computation time: 8150.085938
matmul_shared_kernel GPU Computation time: 8792.204102
matmul_cuda_kernel GPU Computation time: 8755.008789
matmul_opt_kernel GPU Computation time: 11344.499023
```
