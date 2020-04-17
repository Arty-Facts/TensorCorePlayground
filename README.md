# float16
## block size 32

### Coalesing
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
Computing 8192 x 8192 matrix ...
matmul_kernel GPU Computation time: 130927.343750
matmul_shared_kernel GPU Computation time: 61972.496094
matmul_cuda_kernel GPU Computation time: 62078.320313
matmul_opt_kernel GPU Computation time: 60643.042969

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
matmul_kernel GPU Computation time: 13531.789063
matmul_shared_kernel GPU Computation time: 7641.550293
matmul_cuda_kernel GPU Computation time: 7634.952637
matmul_opt_kernel GPU Computation time: 7489.634766

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
Computing 256 x 256 matrix ...
matmul_kernel GPU Computation time: 3.571072
matmul_shared_kernel GPU Computation time: 3.395744
matmul_cuda_kernel GPU Computation time: 3.429952
matmul_opt_kernel GPU Computation time: 3.372160

```


# float32

## block size 32

### Coalesing
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
Computing 8192 x 8192 matrix ...
matmul_kernel GPU Computation time: 12284.616211
matmul_shared_kernel GPU Computation time: 9769.526367
matmul_cuda_kernel GPU Computation time: 9690.458984
matmul_opt_kernel GPU Computation time: 7618.357422


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
matmul_kernel GPU Computation time: 1532.528687
matmul_shared_kernel GPU Computation time: 1205.162354
matmul_cuda_kernel GPU Computation time: 1194.354858
matmul_opt_kernel GPU Computation time: 940.000488

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
Computing 256 x 256 matrix ...
matrixMultiplicationCPU CPU Computation time: 36.329102
matmul_kernel GPU Computation time: 0.764320
Error:  0.000000
matmul_shared_kernel GPU Computation time: 0.594112
Error:  0.000000
matmul_cuda_kernel GPU Computation time: 0.589952
Error:  0.000000
matmul_opt_kernel GPU Computation time: 0.488096
Error:  0.000000

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
matmul_kernel GPU Computation time: 8178.671387
matmul_shared_kernel GPU Computation time: 8610.315430
matmul_cuda_kernel GPU Computation time: 8590.959961
matmul_opt_kernel GPU Computation time: 10774.486328

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
Computing 256 x 256 matrix ...
matrixMultiplicationCPU CPU Computation time: 36.835499
matmul_kernel GPU Computation time: 2.960512
Error:  0.000000
matmul_shared_kernel GPU Computation time: 3.041600
Error:  0.000000
matmul_cuda_kernel GPU Computation time: 3.054592
Error:  0.000000
matmul_opt_kernel GPU Computation time: 3.940448
Error:  0.000000
```
### no Coalesing
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
matmul_kernel GPU Computation time: 891.709961
matmul_shared_kernel GPU Computation time: 1171.492920
matmul_cuda_kernel GPU Computation time: 1161.099121
matmul_opt_kernel GPU Computation time: 903.343933


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
Computing 256 x 256 matrix ...
matrixMultiplicationCPU CPU Computation time: 42.134899
matmul_kernel GPU Computation time: 0.422272
Error:  0.000000
matmul_shared_kernel GPU Computation time: 0.624352
Error:  0.000000
matmul_cuda_kernel GPU Computation time: 0.566656
Error:  0.000000
matmul_opt_kernel GPU Computation time: 0.463296
Error:  0.000000
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
matmul_kernel GPU Computation time: 8112.402344
matmul_shared_kernel GPU Computation time: 8558.845703
matmul_cuda_kernel GPU Computation time: 8571.676758
matmul_opt_kernel GPU Computation time: 10754.331055

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
Computing 256 x 256 matrix ...
matrixMultiplicationCPU CPU Computation time: 40.991299
matmul_kernel GPU Computation time: 3.085344
Error:  0.000000
matmul_shared_kernel GPU Computation time: 3.238912
Error:  0.000000
matmul_cuda_kernel GPU Computation time: 3.054912
Error:  0.000000
matmul_opt_kernel GPU Computation time: 3.814528
Error:  0.000000
```

## block size 16

### Coalesing
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
matmul_kernel GPU Computation time: 868.409851
matmul_shared_kernel GPU Computation time: 1222.726196
matmul_cuda_kernel GPU Computation time: 1214.891235
matmul_opt_kernel GPU Computation time: 953.390076

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
Computing 256 x 256 matrix ...
matrixMultiplicationCPU CPU Computation time: 35.817101
matmul_kernel GPU Computation time: 0.417152
Error:  0.000000
matmul_shared_kernel GPU Computation time: 0.602240
Error:  0.000000
matmul_cuda_kernel GPU Computation time: 0.596032
Error:  0.000000
matmul_opt_kernel GPU Computation time: 0.464352
Error:  0.000000

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
matmul_kernel GPU Computation time: 8229.704102
matmul_shared_kernel GPU Computation time: 8901.735352
matmul_cuda_kernel GPU Computation time: 8865.737305
matmul_opt_kernel GPU Computation time: 11345.245117

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
Computing 256 x 256 matrix ...
matrixMultiplicationCPU CPU Computation time: 36.325100
matmul_kernel GPU Computation time: 2.793472
Error:  0.000000
matmul_shared_kernel GPU Computation time: 2.888064
Error:  0.000000
matmul_cuda_kernel GPU Computation time: 2.958976
Error:  0.000000
matmul_opt_kernel GPU Computation time: 3.619968
Error:  0.000000
```
### no Coalesing

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
Computing 256 x 256 matrix ...
matrixMultiplicationCPU CPU Computation time: 42.227299
matmul_kernel GPU Computation time: 0.406400
Error:  0.000000
matmul_shared_kernel GPU Computation time: 0.617888
Error:  0.000000
matmul_cuda_kernel GPU Computation time: 0.605888
Error:  0.000000
matmul_opt_kernel GPU Computation time: 0.488960
Error:  0.000000

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
Computing 256 x 256 matrix ...
matrixMultiplicationCPU CPU Computation time: 40.650398
matmul_kernel GPU Computation time: 2.762752
Error:  0.000000
matmul_shared_kernel GPU Computation time: 3.054560
Error:  0.000000
matmul_cuda_kernel GPU Computation time: 3.047424
Error:  0.000000
matmul_opt_kernel GPU Computation time: 3.764256
Error:  0.000000
```
