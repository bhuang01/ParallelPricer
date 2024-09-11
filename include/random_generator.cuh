#ifndef RANDOM_GENERATOR_CUH
#define RANDOM_GENERATOR_CUH

#include <curand_kernel.h>

__global__ void init_random_states(curandState *states, unsigned long long seed, int n);
__global__ void generate_uniform_random_numbers(curandState *states, float *random_numbers, int n);
__global__ void generate_normal_random_numbers(curandState *states, float *random_numbers, int n);

cudaError_t generate_random_numbers(float **d_random_numbers, int n, bool use_normal_distribution = false);

#endif