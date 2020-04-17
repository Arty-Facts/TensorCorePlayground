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
double validate(T* val_c, T* c, int N) {
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