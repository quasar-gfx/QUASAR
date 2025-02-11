#ifndef CUDA_IMAGE_H
#define CUDA_IMAGE_H

#if !defined(__APPLE__) && !defined(__ANDROID__)

#include <Texture.h>

#include <cuda_gl_interop.h>
#include <CudaGLInterop/CudaUtils.h>

class CudaGLImage {
public:
    CudaGLImage() = default;
    CudaGLImage(Texture& texture) : texture(&texture) {
        cudautils::checkCudaDevice();
        registerTexture(texture);
    }

    ~CudaGLImage() {
        cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(cudaResource));
    }

    void registerTexture(Texture& texture) {
        this->texture = &texture;
        CHECK_CUDA_ERROR(cudaGraphicsGLRegisterImage(
            &cudaResource,
            texture.ID,
            GL_TEXTURE_2D,
            cudaGraphicsRegisterFlagsReadOnly));
    }

    void map() {
        CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &cudaResource));
    }

    void unmap() {
        CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &cudaResource));
    }

    cudaArray* getArray() {
        cudaArray* array;
        map();
        CHECK_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&array, cudaResource, 0, 0));
        unmap();
        return array;
    }

    void copyToArray(int width, int height, int pitch, void* data) {
        cudaArray* array;
        map();
        CHECK_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&array, cudaResource, 0, 0));
        CHECK_CUDA_ERROR(cudaMemcpy2DFromArray(data, pitch, array, 0, 0, width, height, cudaMemcpyDeviceToHost));
        unmap();
    }

private:
    Texture* texture;

    cudaGraphicsResource* cudaResource;
};
#endif

#endif // CUDA_IMAGE_H
