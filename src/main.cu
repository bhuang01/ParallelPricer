#include <stdio.h>
#include <cuda_runtime.h>
#include "utils.cuh"
#include "random_generator.cuh"
#include "stock_simulation.cuh"

int main() {
    int n = 1000000;
    SimulationParams params = {
        100.0f,
        1.0f,
        0.05f,
        0.2f,
        252
    };
    float strike_price = 100.0f;

    cudaEvent_t start, stop;

    printf("Starting Monte Carlo simulation with %d paths...\n", n);

    float *d_random_numbers;
    start_timer(&start);
    CHECK_CUDA_ERROR(generate_random_numbers(&d_random_numbers, n * params.steps, true));
    float random_gen_time = stop_timer(start, stop);
    printf("Random number generation time: %f ms\n", random_gen_time);

    float *d_stock_paths;
    start_timer(&start);
    CHECK_CUDA_ERROR(run_stock_simulation(d_random_numbers, &d_stock_paths, n, params));
    float simulation_time = stop_timer(start, stop);
    printf("Stock price simulation time: %f ms\n", simulation_time);

    float *d_option_prices;
    CHECK_CUDA_ERROR(cudaMalloc(&d_option_prices, n * sizeof(float)));
    start_timer(&start);
    CHECK_CUDA_ERROR(calculate_option_prices(d_stock_paths, d_option_prices, n, strike_price, true));
    float option_pricing_time = stop_timer(start, stop);
    printf("Option pricing time: %f ms\n", option_pricing_time);

    compute_simulation_statistics(d_stock_paths, d_option_prices, n, params, strike_price);

    CHECK_CUDA_ERROR(cudaFree(d_random_numbers));
    CHECK_CUDA_ERROR(cudaFree(d_stock_paths));
    CHECK_CUDA_ERROR(cudaFree(d_option_prices));

    printf("Monte Carlo simulation completed successfully.\n");

    return 0;
}