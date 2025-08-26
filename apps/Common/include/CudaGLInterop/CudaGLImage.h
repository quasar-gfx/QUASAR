#ifndef CUDA_IMAGE_H
#define CUDA_IMAGE_H

#if defined(HAS_CUDA)

#include <Texture.h>

#include <cuda_gl_interop.h>
#include <CudaGLInterop/CudaUtils.h>

namespace quasar {

class CudaGLImage {
public:
    CudaGLImage() = default;
    CudaGLImage(Texture& texture, cudaStream_t stream = nullptr)
        : texture(&texture)
        , stream(stream)
    {
        cudautils::checkCudaDevice();
        registerTexture(texture);
    }
    ~CudaGLImage() {
        if (ownsStream && stream) cudaStreamSynchronize(stream);
        if (cudaResource) {
            CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(cudaResource));
        }
        if (ownsStream && stream) cudaStreamDestroy(stream);
    }

    void registerTexture(Texture& texture) {
        this->texture = &texture;
        CHECK_CUDA_ERROR(cudaGraphicsGLRegisterImage(
            &cudaResource,
            texture,
            GL_TEXTURE_2D,
            cudaGraphicsRegisterFlagsNone
        ));
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

    cudaArray_t getArrayMapped() {
        cudaArray_t array;
        CHECK_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&array, cudaResource, 0, 0));
        return array;
    }

    void copyArrayToHostAsync(size_t widthBytes, size_t height, size_t dstPitch, void* dstHost) {
        map();
        cudaArray_t array = getArrayMapped();
        CHECK_CUDA_ERROR(cudaMemcpy2DFromArrayAsync(
            dstHost, dstPitch,
            array, 0, 0,
            widthBytes, height,
            cudaMemcpyDeviceToHost,
            stream));
        unmap();
    }

    void copyHostToArrayAsync(size_t widthBytes, size_t height, size_t srcPitch, const void* srcHost) {
        map();
        cudaArray_t array = getArrayMapped();
        CHECK_CUDA_ERROR(cudaMemcpy2DToArrayAsync(
            array, 0, 0,
            srcHost, srcPitch,
            widthBytes, height,
            cudaMemcpyHostToDevice,
            stream));
        unmap();
    }

    void copyArrayToHost(size_t wBytes, size_t h, size_t pitch, void* dstHost) {
        copyArrayToHostAsync(wBytes, h, pitch, dstHost);
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    }

    void copyHostToArray(size_t wBytes, size_t h, size_t pitch, const void* srcHost) {
        copyHostToArrayAsync(wBytes, h, pitch, srcHost);
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
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

    const Texture* texture = nullptr;

    cudaGraphicsResource* cudaResource = nullptr;
    cudaStream_t stream = nullptr;
};

} // namespace quasar

#endif // defined(HAS_CUDA)

#endif // CUDA_IMAGE_H
