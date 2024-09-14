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