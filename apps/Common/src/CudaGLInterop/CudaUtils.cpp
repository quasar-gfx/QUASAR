#if !defined(__APPLE__) && !defined(__ANDROID__)

#include <spdlog/spdlog.h>

#include <CudaGLInterop/CudaUtils.h>

CUdevice cudautils::gDevice = -1;

CUdevice cudautils::checkCudaDevice() {
    if (gDevice != -1) {
        return gDevice;
    }

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        throw std::runtime_error("No CUDA devices found!");
        return -1;
    }

    char name[100];
    cuDeviceGet(&gDevice, 0);
    cuDeviceGetName(name, 100, gDevice);
    spdlog::info("Using CUDA device 0: {}", name);

    return gDevice;
}

#endif
