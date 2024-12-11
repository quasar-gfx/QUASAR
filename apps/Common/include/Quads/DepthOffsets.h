#ifndef DEPTH_OFFSETS_H
#define DEPTH_OFFSETS_H

#include <Texture.h>
#include <Utils/FileIO.h>

#if !defined(__APPLE__) && !defined(__ANDROID__)
#include <filesystem>
#include <cuda_gl_interop.h>
#include <Utils/CudaUtils.h>
#endif

class DepthOffsets {
public:
    glm::uvec2 size;

    Texture buffer;
    std::vector<uint8_t> data;

    DepthOffsets(const glm::uvec2 &size)
        : size(size)
        , buffer({
            .width = size.x,
            .height = size.y,
            .internalFormat = GL_RGBA16F,
            .format = GL_RGBA,
            .type = GL_HALF_FLOAT,
            .wrapS = GL_CLAMP_TO_EDGE,
            .wrapT = GL_CLAMP_TO_EDGE,
            .minFilter = GL_NEAREST,
            .magFilter = GL_NEAREST
        })
        , data(size.x * size.y * 4 * sizeof(uint16_t)) {
#if !defined(__APPLE__) && !defined(__ANDROID__)
    cudautils::checkCudaDevice();
    // register opengl texture with cuda
    CHECK_CUDA_ERROR(cudaGraphicsGLRegisterImage(&cudaResource,
                                                 buffer, GL_TEXTURE_2D,
                                                 cudaGraphicsRegisterFlagsReadOnly));
#endif
    }

    ~DepthOffsets() {
#if !defined(__APPLE__) && !defined(__ANDROID__)
        CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(cudaResource));
#endif
    }

    unsigned int saveToFile(const std::string &filename) {
#if !defined(__APPLE__) && !defined(__ANDROID__)
        cudaArray* cudaBuffer;
        CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &cudaResource));
        CHECK_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&cudaBuffer, cudaResource, 0, 0));
        CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &cudaResource));
        CHECK_CUDA_ERROR(cudaMemcpy2DFromArray(data.data(),
                                                size.x * 4 * sizeof(uint16_t),
                                                cudaBuffer,
                                                0, 0,
                                                size.x * 4 * sizeof(uint16_t), size.y,
                                                cudaMemcpyDeviceToHost));

        std::ofstream file(filename, std::ios::binary);
        file.write(reinterpret_cast<const char*>(data.data()), data.size());
        file.close();
#endif

        return data.size();
    }

    unsigned int loadFromMemory(const char* data) {
#if !defined(__APPLE__) && !defined(__ANDROID__)
        cudaArray* cudaBuffer;
        CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &cudaResource));
        CHECK_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&cudaBuffer, cudaResource, 0, 0));
        CHECK_CUDA_ERROR(cudaMemcpy2DToArray(cudaBuffer, 0, 0,
                                             data, size.x * 4 * sizeof(uint16_t),
                                             size.x * 4 * sizeof(uint16_t), size.y,
                                             cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &cudaResource));
#else
        buffer.setData(size.x, size.y, data);
#endif
        return this->data.size();
    }

    unsigned int loadFromFile(const std::string &filename) {
#if !defined(__APPLE__) && !defined(__ANDROID__)
        if (!std::filesystem::exists(filename)) {
            std::cerr << "File " << filename << " does not exist" << std::endl;
            return 0;
        }
#endif
        auto depthOffsets = FileIO::loadBinaryFile(filename);
        return loadFromMemory(depthOffsets.data());
    }

private:
#if !defined(__APPLE__) && !defined(__ANDROID__)
    cudaGraphicsResource* cudaResource;
#endif
};

#endif // DEPTH_OFFSETS_H
