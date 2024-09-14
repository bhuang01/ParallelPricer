#include "random_generator.cuh"
#include "utils.cuh"

__global__ void init_random_states(curandState *states, unsigned long long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void generate_uniform_random_numbers(curandState *states, float *random_numbers, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        random_numbers[idx] = curand_uniform(&states[idx]);
    }
}

__global__ void generate_normal_random_numbers(curandState *states, float *random_numbers, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        random_numbers[idx] = curand_normal(&states[idx]);
    }
}

cudaError_t generate_random_numbers(float **d_random_numbers, int n, bool use_normal_distribution) {
    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc(d_random_numbers, n * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaStatus;
    }

    curandState *d_states;
    cudaStatus = cudaMalloc(&d_states, n * sizeof(curandState));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        cudaFree(*d_random_numbers);
        return cudaStatus;
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    init_random_states<<<blocksPerGrid, threadsPerBlock>>>(d_states, time(NULL), n);

    if (use_normal_distribution) {
        generate_normal_random_numbers<<<blocksPerGrid, threadsPerBlock>>>(d_states, *d_random_numbers, n);
    } else {
        generate_uniform_random_numbers<<<blocksPerGrid, threadsPerBlock>>>(d_states, *d_random_numbers, n);
    }

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(*d_random_numbers);
        cudaFree(d_states);
        return cudaStatus;
    }

    cudaFree(d_states);

    return cudaStatus;
}