#ifndef CUDA_BUFFER_H
#define CUDA_BUFFER_H

#if defined(HAS_CUDA)

#include <Buffer.h>

#include <cuda_gl_interop.h>
#include <CudaGLInterop/CudaUtils.h>

namespace quasar {

class CudaGLBuffer {
public:
    CudaGLBuffer() = default;
    CudaGLBuffer(Buffer& buffer, cudaStream_t stream = nullptr)
        : buffer(&buffer)
        , stream(stream)
    {
        cudautils::checkCudaDevice();
        registerBuffer(buffer);
    }
    ~CudaGLBuffer() {
        if (ownsStream && stream) cudaStreamSynchronize(stream);
        if (cudaResource) {
            CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(cudaResource));
        }
        if (ownsStream && stream) cudaStreamDestroy(stream);
    }

    void registerBuffer(Buffer& buffer) {
        this->buffer = &buffer;
        CHECK_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(
            &cudaResource,
            buffer,
            cudaGraphicsRegisterFlagsNone));
    }

    void ensureStream() {
        if (!stream) {
            CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
            ownsStream = true;
        }
    }

    void map() {
        ensureStream();
        CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &cudaResource, stream));
    }

    void unmap() {
        CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &cudaResource, stream));
    }

    void* getPtr() {
        void* cudaPtr; size_t size;
        map();
        CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(&cudaPtr, &size, cudaResource));
        unmap();
        return cudaPtr;
    }

    void copyFromHostAsync(const void* src, size_t bytes) {
        ensureStream();
        CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &cudaResource, stream));
        void* dst = nullptr; size_t size = 0;
        CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(&dst, &size, cudaResource));
        // Clamp to mapped size (defensive; keep behavior predictable)
        const size_t n = (bytes <= size) ? bytes : size;
        CHECK_CUDA_ERROR(cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &cudaResource, stream));
    }

    void copyFromDeviceAsync(const void* srcDevice, size_t bytes) {
        ensureStream();
        CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &cudaResource, stream));
        void* dst = nullptr; size_t size = 0;
        CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(&dst, &size, cudaResource));
        const size_t n = (bytes <= size) ? bytes : size;
        CHECK_CUDA_ERROR(cudaMemcpyAsync(dst, srcDevice, n, cudaMemcpyDeviceToDevice, stream));
        CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &cudaResource, stream));
    }

    void copyToHostAsync(void* dstHost, size_t bytes) {
        ensureStream();
        CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &cudaResource, stream));
        void* src = nullptr; size_t size = 0;
        CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(&src, &size, cudaResource));
        const size_t n = (bytes <= size) ? bytes : size;
        CHECK_CUDA_ERROR(cudaMemcpyAsync(dstHost, src, n, cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &cudaResource, stream));
    }

    void synchronize() {
        ensureStream();
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    }

    cudaStream_t getStream() const { return stream; }
    void setStream(cudaStream_t s) { stream = s; ownsStream = false; }

    static void registerHostBuffer(void* ptr, size_t size) {
        CHECK_CUDA_ERROR(cudaHostRegister(ptr, size, 0));
    }

    static void unregisterHostBuffer(void* ptr) {
        CHECK_CUDA_ERROR(cudaHostUnregister(ptr));
    }

private:
    bool ownsStream = false;

    const Buffer* buffer = nullptr;

    cudaGraphicsResource* cudaResource = nullptr;
    cudaStream_t stream = nullptr;
};

} // namespace quasar

#endif // defined(HAS_CUDA)

#endif // CUDA_BUFFER_H
