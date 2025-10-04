#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#if defined(QUASAR_HAS_CUDA)

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

#endif

#endif // CUDA_UTILS_H
