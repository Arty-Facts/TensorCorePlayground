template<typename T>
float matrixMultiplicationCPU(T* c, const T* a, const T* b, unsigned int N) {
    float sum;
    auto start_cpu_time = std::chrono::high_resolution_clock::now();
    for (unsigned int row = 0; row < N; row++) {
        for (unsigned int col = 0; col < N; col++) {
            sum = 0.f;
            for (unsigned int n = 0; n < N; n++) {
                //sum += (float)a[row * N + n] * (float)b[n * N + col];
                //assuming the matrix is transposed for better Coalescing
                sum += (float)a[row * N + n] * (float)b[col * N + n];
            }
            c[row * N + col] = (T)sum;
        }
    }
    auto end_cpu_time = std::chrono::high_resolution_clock::now();
    float cpu_time = (float)std::chrono::duration<double, std::milli>(end_cpu_time - start_cpu_time).count();
    return cpu_time;

}
template<typename T>
double validate(T* val_c, float* c, int N) {
    double err = 0;
    // Check the result and make sure it is correct
    for (int ROW = 0; ROW < N; ROW++) {
        for (int COL = 0; COL < N; COL++) {
            //printf("cpu: %f -> gpu: %f\n", val_c[ROW * N + COL], c[ROW * N + COL]);
            err += (double)val_c[ROW * N + COL] - (double)c[ROW * N + COL];
        }
    }
    return err;
}