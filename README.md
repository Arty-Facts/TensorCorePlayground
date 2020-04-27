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
Type size 2
matmul_kernel GPU Computation time: 6058.397461
TFLOPS: 0.18
Error:  380.932135
matmul_shared_kernel GPU Computation time: 1284.902832
TFLOPS: 0.86
Error:  -3482.505853
matmul_cuda_kernel GPU Computation time: 1236.101074
TFLOPS: 0.89
Error:  -3482.505853
matmul_opt_kernel GPU Computation time: 731.648438
TFLOPS: 1.50
Error:  380.932135
matmul_mma_kernel GPU Computation time: 185.267548
TFLOPS: 5.93
Error:  202.408698

Computing 4096 x 4096 matrix ...
Type size 2
matmul_kernel GPU Computation time: 759.011536
TFLOPS: 0.18
Error:  -112.840082
matmul_shared_kernel GPU Computation time: 161.027908
TFLOPS: 0.85
Error:  -163.299066
matmul_cuda_kernel GPU Computation time: 154.923431
TFLOPS: 0.89
Error:  -163.299066
matmul_opt_kernel GPU Computation time: 91.819389
TFLOPS: 1.50
Error:  -112.840082
matmul_mma_kernel GPU Computation time: 23.293089
TFLOPS: 5.90
Error:  116.289801

Computing 1024 x 1024 matrix ...
Type size 2
matrixMultiplicationCPU CPU Computation time: 881.934326
TFLOPS: 0.00
matmul_kernel GPU Computation time: 14.686048
TFLOPS: 0.15
Error:  -29.394117
matmul_shared_kernel GPU Computation time: 3.016352
TFLOPS: 0.71
Error:  -49.303297
matmul_cuda_kernel GPU Computation time: 3.532512
TFLOPS: 0.61
Error:  -49.303297
matmul_opt_kernel GPU Computation time: 1.790656
TFLOPS: 1.20
Error:  -29.394117
matmul_mma_kernel GPU Computation time: 0.636832
TFLOPS: 3.37
Error:  -25.666578

Computing 512 x 512 matrix ...
Type size 2
matrixMultiplicationCPU CPU Computation time: 107.120903
TFLOPS: 0.00
matmul_kernel GPU Computation time: 2.056960
TFLOPS: 0.13
Error:  -11.940039
matmul_shared_kernel GPU Computation time: 0.463904
TFLOPS: 0.58
Error:  -26.119726
matmul_cuda_kernel GPU Computation time: 0.464768
TFLOPS: 0.58
Error:  -26.119726
matmul_opt_kernel GPU Computation time: 0.321344
TFLOPS: 0.84
Error:  -11.940039
matmul_mma_kernel GPU Computation time: 0.131328
TFLOPS: 2.04
Error:  -11.420325

Computing 256 x 256 matrix ...
Type size 2
matrixMultiplicationCPU CPU Computation time: 12.340800
TFLOPS: 0.00
matmul_kernel GPU Computation time: 0.470848
TFLOPS: 0.07
Error:  -5.419451
matmul_shared_kernel GPU Computation time: 0.202208
TFLOPS: 0.17
Error:  0.665998
matmul_cuda_kernel GPU Computation time: 0.167424
TFLOPS: 0.20
Error:  0.665998
matmul_opt_kernel GPU Computation time: 0.168000
TFLOPS: 0.20
Error:  -5.419451
matmul_mma_kernel GPU Computation time: 0.112640
TFLOPS: 0.30
Error:  -2.144549

Computing 128 x 128 matrix ...
Type size 2
matrixMultiplicationCPU CPU Computation time: 1.398200
TFLOPS: 0.00
matmul_kernel GPU Computation time: 0.127264
TFLOPS: 0.03
Error:  1.222947
matmul_shared_kernel GPU Computation time: 0.078016
TFLOPS: 0.05
Error:  0.941697
matmul_cuda_kernel GPU Computation time: 0.084096
TFLOPS: 0.05
Error:  0.941697
matmul_opt_kernel GPU Computation time: 0.077504
TFLOPS: 0.05
Error:  1.222947
matmul_mma_kernel GPU Computation time: 0.071200
TFLOPS: 0.06
Error:  0.520798

Computing 64 x 64 matrix ...
Type size 2
matrixMultiplicationCPU CPU Computation time: 0.145900
TFLOPS: 0.00
matmul_kernel GPU Computation time: 0.111040
TFLOPS: 0.00
Error:  -0.209815
matmul_shared_kernel GPU Computation time: 0.070720
TFLOPS: 0.01
Error:  0.422021
matmul_cuda_kernel GPU Computation time: 0.079328
TFLOPS: 0.01
Error:  0.422021
matmul_opt_kernel GPU Computation time: 0.069664
TFLOPS: 0.01
Error:  -0.209815
matmul_mma_kernel GPU Computation time: 0.063264
TFLOPS: 0.01
Error:  0.392236

Computing 32 x 32 matrix ...
Type size 2
matrixMultiplicationCPU CPU Computation time: 0.017300
TFLOPS: 0.00
matmul_kernel GPU Computation time: 0.091968
TFLOPS: 0.00
Error:  0.100657
matmul_shared_kernel GPU Computation time: 0.087904
TFLOPS: 0.00
Error:  0.059764
matmul_cuda_kernel GPU Computation time: 0.075104
TFLOPS: 0.00
Error:  0.059764
matmul_opt_kernel GPU Computation time: 0.083680
TFLOPS: 0.00
Error:  0.100657
matmul_mma_kernel GPU Computation time: 0.065600
TFLOPS: 0.00
Error:  0.096751

Computing 16 x 16 matrix ...
Type size 2
matrixMultiplicationCPU CPU Computation time: 0.002200
TFLOPS: 0.00
matmul_kernel GPU Computation time: 0.340064
TFLOPS: 0.00
Error:  -0.031373
matmul_shared_kernel GPU Computation time: 0.079552
TFLOPS: 0.00
Error:  0.034056
matmul_cuda_kernel GPU Computation time: 0.061472
TFLOPS: 0.00
Error:  0.034056
matmul_opt_kernel GPU Computation time: 0.080192
TFLOPS: 0.00
Error:  -0.031373
matmul_mma_kernel GPU Computation time: 0.059392
TFLOPS: 0.00
Error:  -0.005983

Computing 8 x 8 matrix ...
Type size 2
matrixMultiplicationCPU CPU Computation time: 0.000300
TFLOPS: 0.00
matmul_kernel GPU Computation time: 0.081056
TFLOPS: 0.00
Error:  0.001344
matmul_shared_kernel GPU Computation time: 0.061440
TFLOPS: 0.00
Error:  -0.000609
matmul_cuda_kernel GPU Computation time: 0.070112
TFLOPS: 0.00
Error:  -0.000609
matmul_opt_kernel GPU Computation time: 0.200576
TFLOPS: 0.00
Error:  0.001344
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
Type size 4
matmul_kernel GPU Computation time: 12128.789063
TFLOPS: 0.09
Error:  0.000000
matmul_shared_kernel GPU Computation time: 1275.279785
TFLOPS: 0.86
Error:  0.000000
matmul_cuda_kernel GPU Computation time: 1063.254761
TFLOPS: 1.03
Error:  0.000000
matmul_opt_kernel GPU Computation time: 1283.370972
TFLOPS: 0.86
Error:  0.000000

Computing 4096 x 4096 matrix ...
Type size 4
matmul_kernel GPU Computation time: 1517.678711
TFLOPS: 0.09
Error:  0.000000
matmul_shared_kernel GPU Computation time: 158.981766
TFLOPS: 0.86
Error:  0.000000
matmul_cuda_kernel GPU Computation time: 132.926437
TFLOPS: 1.03
Error:  0.000000
matmul_opt_kernel GPU Computation time: 160.064026
TFLOPS: 0.86
Error:  0.000000

Computing 1024 x 1024 matrix ...
Type size 4
matrixMultiplicationCPU CPU Computation time: 883.318787
TFLOPS: 0.00
matmul_kernel GPU Computation time: 29.451937
TFLOPS: 0.07
Error:  0.033912
matmul_shared_kernel GPU Computation time: 3.090688
TFLOPS: 0.69
Error:  0.033912
matmul_cuda_kernel GPU Computation time: 2.604768
TFLOPS: 0.82
Error:  0.033912
matmul_opt_kernel GPU Computation time: 3.150976
TFLOPS: 0.68
Error:  0.033912

Computing 512 x 512 matrix ...
Type size 4
matrixMultiplicationCPU CPU Computation time: 108.858498
TFLOPS: 0.00
matmul_kernel GPU Computation time: 4.047360
TFLOPS: 0.07
Error:  0.003528
matmul_shared_kernel GPU Computation time: 0.527232
TFLOPS: 0.51
Error:  0.003528
matmul_cuda_kernel GPU Computation time: 0.489088
TFLOPS: 0.55
Error:  0.003528
matmul_opt_kernel GPU Computation time: 0.522240
TFLOPS: 0.51
Error:  0.003528

 
Computing 256 x 256 matrix ...
Type size 4
matrixMultiplicationCPU CPU Computation time: 13.102700
TFLOPS: 0.00
matmul_kernel GPU Computation time: 0.747584
TFLOPS: 0.04
Error:  -0.001167
matmul_shared_kernel GPU Computation time: 0.132352
TFLOPS: 0.25
Error:  -0.001167
matmul_cuda_kernel GPU Computation time: 0.120032
TFLOPS: 0.28
Error:  -0.001167
matmul_opt_kernel GPU Computation time: 0.198112
TFLOPS: 0.17
Error:  -0.001167

Computing 128 x 128 matrix ...
Type size 4
matrixMultiplicationCPU CPU Computation time: 1.394900
TFLOPS: 0.00
matmul_kernel GPU Computation time: 0.180768
TFLOPS: 0.02
Error:  -0.000199
matmul_shared_kernel GPU Computation time: 0.114144
TFLOPS: 0.04
Error:  -0.000199
matmul_cuda_kernel GPU Computation time: 0.079808
TFLOPS: 0.05
Error:  -0.000199
matmul_opt_kernel GPU Computation time: 0.083776
TFLOPS: 0.05
Error:  -0.000199

Computing 64 x 64 matrix ...
Type size 4
matrixMultiplicationCPU CPU Computation time: 0.149700
TFLOPS: 0.00
matmul_kernel GPU Computation time: 0.187328
TFLOPS: 0.00
Error:  -0.000013
matmul_shared_kernel GPU Computation time: 0.118848
TFLOPS: 0.00
Error:  -0.000013
matmul_cuda_kernel GPU Computation time: 0.121952
TFLOPS: 0.00
Error:  -0.000013
matmul_opt_kernel GPU Computation time: 0.098400
TFLOPS: 0.01
Error:  -0.000013

Computing 32 x 32 matrix ...
Type size 4
matrixMultiplicationCPU CPU Computation time: 0.017600
TFLOPS: 0.00
matmul_kernel GPU Computation time: 0.165984
TFLOPS: 0.00
Error:  0.000006
matmul_shared_kernel GPU Computation time: 0.107616
TFLOPS: 0.00
Error:  0.000006
matmul_cuda_kernel GPU Computation time: 0.063264
TFLOPS: 0.00
Error:  0.000006
matmul_opt_kernel GPU Computation time: 0.113376
TFLOPS: 0.00
Error:  0.000006

Computing 16 x 16 matrix ...
Type size 4
matrixMultiplicationCPU CPU Computation time: 0.002100
TFLOPS: 0.00
matmul_kernel GPU Computation time: 0.146272
TFLOPS: 0.00
Error:  0.000009
matmul_shared_kernel GPU Computation time: 0.112256
TFLOPS: 0.00
Error:  0.000009
matmul_cuda_kernel GPU Computation time: 0.085248
TFLOPS: 0.00
Error:  0.000009
matmul_opt_kernel GPU Computation time: 0.063360
TFLOPS: 0.00
Error:  0.000009

Computing 8 x 8 matrix ...
Type size 4
matrixMultiplicationCPU CPU Computation time: 0.000300
TFLOPS: 0.00
matmul_kernel GPU Computation time: 0.200736
TFLOPS: 0.00
Error:  0.000003
matmul_shared_kernel GPU Computation time: 0.120864
TFLOPS: 0.00
Error:  0.000003
matmul_cuda_kernel GPU Computation time: 0.094432
TFLOPS: 0.00
Error:  0.000003
matmul_opt_kernel GPU Computation time: 0.088096
TFLOPS: 0.00
Error:  0.000003

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
