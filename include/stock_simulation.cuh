#ifndef STOCK_SIMULATION_CUH
#define STOCK_SIMULATION_CUH

#include <cuda_runtime.h>

struct SimulationParams {
    float S0;
    float T;
    float r;
    float sigma;
    int steps;
};

__global__ void simulate_stock_paths(float* d_random_numbers, float* d_stock_paths, int n, SimulationParams params);

cudaError_t run_stock_simulation(float* d_random_numbers, float** d_stock_paths, int n, SimulationParams params);

cudaError_t calculate_option_prices(float* d_stock_paths, float* d_option_prices, int n, float strike_price, bool is_call);

void compute_simulation_statistics(float* d_stock_paths, float* d_option_prices, int n, SimulationParams params, float strike_price);

#endif