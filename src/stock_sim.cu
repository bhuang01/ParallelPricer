#include <curand_kernel.h>
#include <stdio.h>
#include <math.h>
#include "stock_simulation.cuh"
#include "utils.cuh"

__global__ void simulate_stock_paths(float* d_random_numbers, float* d_stock_paths, int n, SimulationParams params) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float dt = params.T / params.steps;
        float S = params.S0;
        for (int i = 0; i < params.steps; i++) {
            float dW = sqrt(dt) * d_random_numbers[idx * params.steps + i];
            S = S * exp((params.r - 0.5 * params.sigma * params.sigma) * dt + params.sigma * dW);
        }
        d_stock_paths[idx] = S;
    }
}

cudaError_t run_stock_simulation(float* d_random_numbers, float** d_stock_paths, int n, SimulationParams params) {
    cudaError_t cudaStatus;

    // Allocate device memory for stock paths
    cudaStatus = cudaMalloc(d_stock_paths, n * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for stock paths!");
        return cudaStatus;
    }

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    simulate_stock_paths<<<blocksPerGrid, threadsPerBlock>>>(d_random_numbers, *d_stock_paths, n, params);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "simulate_stock_paths launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }

    return cudaStatus;
}

__global__ void calculate_option_prices_kernel(float* d_stock_paths, float* d_option_prices, int n, float strike_price, bool is_call) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float stock_price = d_stock_paths[idx];
        float payoff;
        if (is_call) {
            payoff = fmaxf(stock_price - strike_price, 0.0f);
        } else {
            payoff = fmaxf(strike_price - stock_price, 0.0f);
        }
        d_option_prices[idx] = payoff;
    }
}

cudaError_t calculate_option_prices(float* d_stock_paths, float* d_option_prices, int n, float strike_price, bool is_call) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    calculate_option_prices_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_stock_paths, d_option_prices, n, strike_price, is_call);

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "calculate_option_prices_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    return cudaStatus;
}

__global__ void reduce_sum(float* d_data, float* d_result, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? d_data[i] : 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) d_result[blockIdx.x] = sdata[0];
}

void compute_simulation_statistics(float* d_stock_paths, float* d_option_prices, int n, SimulationParams params, float strike_price) {
    float *d_temp, *h_temp;
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    cudaMalloc(&d_temp, blocksPerGrid * sizeof(float));
    h_temp = (float*)malloc(blocksPerGrid * sizeof(float));

    // Compute average stock price
    reduce_sum<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_stock_paths, d_temp, n);
    cudaMemcpy(h_temp, d_temp, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    float avg_stock_price = 0;
    for (int i = 0; i < blocksPerGrid; i++) {
        avg_stock_price += h_temp[i];
    }
    avg_stock_price /= n;

    // Compute average option price
    reduce_sum<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_option_prices, d_temp, n);
    cudaMemcpy(h_temp, d_temp, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    float avg_option_price = 0;
    for (int i = 0; i < blocksPerGrid; i++) {
        avg_option_price += h_temp[i];
    }
    avg_option_price /= n;

    // Discount the option price
    float discounted_option_price = avg_option_price * exp(-params.r * params.T);

    printf("Simulation Results:\n");
    printf("Average Stock Price: %.4f\n", avg_stock_price);
    printf("Average Option Price: %.4f\n", avg_option_price);
    printf("Discounted Option Price: %.4f\n", discounted_option_price);

    cudaFree(d_temp);
    free(h_temp);
}