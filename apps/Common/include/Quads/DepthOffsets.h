#ifndef DEPTH_OFFSETS_H
#define DEPTH_OFFSETS_H

#include <Path.h>
#include <Buffer.h>
#include <Texture.h>

#if defined(HAS_CUDA)
#include <CudaGLInterop/CudaGLImage.h>
#endif

namespace quasar {

class DepthOffsets {
public:
    Texture texture;

    DepthOffsets(const glm::uvec2& textureSize);
    ~DepthOffsets() = default;

    const glm::uvec2& getSize() const {
        return textureSize;
    }

#ifdef GL_CORE
    size_t copyToCPU(std::vector<char>& outputData);
#endif
    size_t copyFromCPU(std::vector<char>& inputData);

private:
    glm::uvec2 textureSize;

    Buffer uploadPBO;
#if defined(HAS_CUDA)
    CudaGLImage cudaImage;
#endif

    static inline size_t bytesPerRow(unsigned w) {
        return static_cast<size_t>(w) * 4 * sizeof(uint16_t); // 4 channels * 16 bit floats
    }
};

} // namespace quasar

#endif // DEPTH_OFFSETS_H
