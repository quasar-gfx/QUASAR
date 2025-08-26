#include <spdlog/spdlog.h>

#include <Quads/DepthOffsets.h>

using namespace quasar;

DepthOffsets::DepthOffsets(const glm::uvec2& textureSize)
    : textureSize(textureSize)
    , texture({
        .width = textureSize.x,
        .height = textureSize.y,
        .internalFormat = GL_RGBA16F, // 4 offsets per subpixel corner
        .format = GL_RGBA,
        .type = GL_HALF_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    })
    , uploadPBO({
        .target = GL_PIXEL_UNPACK_BUFFER,
        .dataSize = sizeof(uint16_t),
        .numElems = textureSize.x * textureSize.y * 4, // 4 channels
        .usage = GL_STREAM_DRAW
    })
#if defined(HAS_CUDA)
    , cudaImage(texture)
#endif
{}

#ifdef GL_CORE
size_t DepthOffsets::copyToCPU(std::vector<char>& outputData) {
#if defined(HAS_CUDA)
    size_t rowBytes = bytesPerRow(textureSize.x);
    size_t outputSize = rowBytes * textureSize.y;
    outputData.resize(outputSize);

    CudaGLImage::registerHostBuffer(outputData.data(), outputSize);
    cudaImage.copyArrayToHostAsync(rowBytes, textureSize.y, rowBytes, outputData.data());
    cudaImage.synchronize();
    CudaGLImage::unregisterHostBuffer(outputData.data());
#else
    size_t rowBytes = bytesPerRow(textureSize.x);
    size_t outputSize = rowBytes * textureSize.y;
    outputData.resize(outputSize);

    uploadPBO.bind();
    glBufferData(GL_PIXEL_PACK_BUFFER, outputSize, nullptr, GL_STREAM_READ);

    texture.bind();
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_HALF_FLOAT, nullptr);

    uploadPBO.getData(outputData.data());
    uploadPBO.unbind();
#endif

    return outputData.size();
}
#endif

size_t DepthOffsets::copyFromCPU(std::vector<char>& inputData) {
    uploadPBO.bind();

    // Map, copy, unmap
    void* dst = uploadPBO.mapToCPU(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_UNSYNCHRONIZED_BIT);
    if (dst) {
        std::memcpy(dst, inputData.data(), inputData.size());
        uploadPBO.unmapFromCPU();

        // Upload texture data
        glPixelStorei(GL_UNPACK_ALIGNMENT, texture.alignment);
        texture.bind();
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, textureSize.x, textureSize.y, GL_RGBA, GL_HALF_FLOAT, nullptr);
    }
    else {
        spdlog::warn("Failed to map depthOffsets PBO. Copying using loadFromData");
        texture.loadFromData(inputData.data());
    }

    uploadPBO.unbind();
    return inputData.size();
}
