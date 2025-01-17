#ifndef DEPTH_OFFSETS_H
#define DEPTH_OFFSETS_H

#include <spdlog/spdlog.h>
#include <lz4.h>
#include <lz4_stream/lz4_stream.h>

#include <Texture.h>
#include <Utils/FileIO.h>

#if !defined(__ANDROID__)
#include <filesystem>
#endif

#if !defined(__APPLE__) && !defined(__ANDROID__)
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

        // setup LZ4 decompression context
        auto status = LZ4F_createDecompressionContext(&dctx, LZ4F_VERSION);
        if (LZ4F_isError(status)) {
            spdlog::error("Failed to create LZ4 context: {}", LZ4F_getErrorName(status));
            return;
        }
    }

    ~DepthOffsets() {
#if !defined(__APPLE__) && !defined(__ANDROID__)
        CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(cudaResource));
#endif

        LZ4F_freeDecompressionContext(dctx);
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

    unsigned int loadFromFile(const std::string &filename, unsigned int* numBytesLoaded = nullptr) {
#if !defined(__ANDROID__)
        if (!std::filesystem::exists(filename)) {
            spdlog::error("File {} does not exist", filename);
            return 0;
        }
#endif

        auto depthOffsetsCompressed = FileIO::loadBinaryFile(filename);
        size_t srcSize = depthOffsetsCompressed.size();
        size_t dstSize = data.size();

        auto ret = LZ4F_decompress(dctx,
                                   data.data(), &dstSize,
                                   depthOffsetsCompressed.data(), &srcSize,
                                   nullptr);
        if (LZ4F_isError(ret)) {
            spdlog::error("LZ4 decompression failed: {}", LZ4F_getErrorName(ret));
            return 0;
        }

        if (numBytesLoaded != nullptr) {
            *numBytesLoaded = srcSize;
        }

        return loadFromMemory(reinterpret_cast<const char*>(data.data()));
    }

#if !defined(__APPLE__) && !defined(__ANDROID__)
    unsigned int saveToMemory(std::vector<char> &compressedData) {
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

        compressedData.resize(LZ4F_compressFrameBound(data.size(), nullptr));
        int outputSize = LZ4_compress_default(
                            reinterpret_cast<const char*>(data.data()),
                            compressedData.data(),
                            data.size(),
                            compressedData.size());
        return outputSize;
    }

    unsigned int saveToFile(const std::string &filename) {
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

        std::ofstream quadsFile(filename + ".lz4", std::ios::binary);
        lz4_stream::ostream lz4_stream(quadsFile);
        lz4_stream.write(reinterpret_cast<const char*>(data.data()), data.size());
        lz4_stream.close();

        return data.size();
    }
#endif

private:
    LZ4F_dctx* dctx = nullptr;

#if !defined(__APPLE__) && !defined(__ANDROID__)
    cudaGraphicsResource* cudaResource;
#endif
};

#endif // DEPTH_OFFSETS_H
