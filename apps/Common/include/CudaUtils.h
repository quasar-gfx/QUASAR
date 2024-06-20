#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#ifndef __APPLE__

#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

namespace CudaUtils {

extern CUdevice gDevice;

CUdevice findCudaDevice();

}
#endif

#endif // CUDA_UTILS_H
