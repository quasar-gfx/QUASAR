#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#ifndef __APPLE__

#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(call) do {                                         \
    cudaError_t err = call;                                                 \
    if (cudaSuccess != err) {                                               \
        fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                   \
                __FUNCTION__, __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                                 \
    } } while(0)

namespace CudaUtils {

extern CUdevice gDevice;

CUdevice findCudaDevice();

}
#endif

#endif // CUDA_UTILS_H
