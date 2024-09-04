#include <curand_kernel.h>
#include <stdio.h>

__global__ void simulate_stock_paths(float* d_random_numbers, float* d_stock_paths, int n, float S0, float T, float r, float sigma, int steps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float dt = T / steps;
        float S = S0;
        for (int i = 0; i < steps; i++) {
            float dW = sqrt(dt) * d_random_numbers[idx * steps + i];
            S = S * exp((r - 0.5 * sigma * sigma) * dt + sigma * dW);
        }
        d_stock_paths[idx] = S;
    }
}

int main() {
    // params
    int n = 1000;
    int steps = 100;
    float S0 = 100.0;
    float T = 1.0;
    float r = 0.05;
    float sigma = 0.2;

    // memory
    float* d_random_numbers, * d_stock_paths;
    cudaMalloc(&d_random_numbers, n * steps * sizeof(float));
    cudaMalloc(&d_stock_paths, n * sizeof(float));

    // rand number generator
    generate_random_numbers << <(n * steps + 255) / 256, 256 >> > (d_random_numbers, n * steps);
    cudaDeviceSynchronize();

    // simulate stock path
    simulate_stock_paths << <(n + 255) / 256, 256 >> > (d_random_numbers, d_stock_paths, n, S0, T, r, sigma, steps);
    cudaDeviceSynchronize();
    float* h_stock_paths = (float*)malloc(n * sizeof(float));
    cudaMemcpy(h_stock_paths, d_stock_paths, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; i++) {
        printf("%f\n", h_stock_paths[i]);
    }

    cudaFree(d_random_numbers);
    cudaFree(d_stock_paths);
    free(h_stock_paths);
    return 0;
}
