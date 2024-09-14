#ifndef UTILS_CUH
#define UTILS_CUH

#include <stdio.h>
#include <cuda_runtime.h>

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line);

#define CHECK_CUDA_ERROR(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void start_timer(cudaEvent_t *start, cudaEvent_t *stop);
float stop_timer(cudaEvent_t start, cudaEvent_t stop);

#endif