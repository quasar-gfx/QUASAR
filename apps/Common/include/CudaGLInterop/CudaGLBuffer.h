#ifndef CUDA_BUFFER_H
#define CUDA_BUFFER_H

#if !defined(__APPLE__) && !defined(__ANDROID__)

#include <Buffer.h>

#include <cuda_gl_interop.h>
#include <CudaGLInterop/CudaUtils.h>

namespace quasar {

class CudaGLBuffer {
public:
    CudaGLBuffer() = default;
    CudaGLBuffer(Buffer& buffer) : buffer(&buffer) {
        cudautils::checkCudaDevice();
        registerBuffer(buffer);
    }

    ~CudaGLBuffer() {
        cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(cudaResource));
    }

    void registerBuffer(Buffer& buffer) {
        this->buffer = &buffer;
        CHECK_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(
            &cudaResource,
            buffer,
            cudaGraphicsRegisterFlagsNone));
    }

    void map() {
        CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &cudaResource));
    }

    void unmap() {
        CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &cudaResource));
    }

    void* getPtr() {
        void* cudaPtr; size_t size;
        map();
        CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(&cudaPtr, &size, cudaResource));
        unmap();
        return cudaPtr;
    }

private:
    Buffer* buffer;

    cudaGraphicsResource* cudaResource;
};
#endif

} // namespace quasar

#endif // CUDA_BUFFER_H
