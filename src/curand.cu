#include <stdio.h>
#include <curand_kernel.h>

__global__ void generate_random_numbers(float *random_numbers, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(1234, idx, 0, &state); // init
    if (idx < n) {
        random_numbers[idx] = curand_uniform(&state);
    }
}

int main() {
    int n = 1000;
    float *d_random_numbers;
    cudaMalloc(&d_random_numbers, n * sizeof(float));
    generate_random_numbers<<<(n + 255) / 256, 256>>>(d_random_numbers, n);
    cudaDeviceSynchronize();

    // Copy data back to host and print for verification
    float *h_random_numbers = (float *)malloc(n * sizeof(float));
    cudaMemcpy(h_random_numbers, d_random_numbers, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; i++) {
        printf("%f\n", h_random_numbers[i]);
    }

    cudaFree(d_random_numbers);
    free(h_random_numbers);
    return 0;
}
