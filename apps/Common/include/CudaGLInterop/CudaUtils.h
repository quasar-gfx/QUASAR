#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#if !defined(__APPLE__) && !defined(__ANDROID__)

#include <spdlog/spdlog.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace quasar {

#define CHECK_CUDA_ERROR(call) do {                                 \
    cudaError_t err = call;                                         \
    if (cudaSuccess != err) {                                       \
        spdlog::error("CUDA error in file '{}' in line {}: {}",     \
                      __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                         \
    } } while(0)

namespace cudautils {

extern CUdevice gDevice;

CUdevice checkCudaDevice();

}

} // namespace quasar

#endif // !defined(__APPLE__) && !defined(__ANDROID__)

#endif // CUDA_UTILS_H
