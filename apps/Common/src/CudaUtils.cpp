#ifndef __APPLE__

#include <Utils/CudaUtils.h>

CUdevice cudautils::gDevice = -1;

CUdevice cudautils::findCudaDevice() {
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
    std::cout << "Using CUDA Device 0: " << name << std::endl;

    return gDevice;
}

#endif
