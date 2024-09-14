#include "utils.cuh"

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), cudaGetErrorString(result), func);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

void start_timer(cudaEvent_t *start, cudaEvent_t *stop) {
    CHECK_CUDA_ERROR(cudaEventCreate(start));
    CHECK_CUDA_ERROR(cudaEventCreate(stop));
    CHECK_CUDA_ERROR(cudaEventRecord(*start));
}

float stop_timer(cudaEvent_t start, cudaEvent_t stop) {
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    return milliseconds;
}