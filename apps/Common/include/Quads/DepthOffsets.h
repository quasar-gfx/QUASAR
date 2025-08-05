#ifndef DEPTH_OFFSETS_H
#define DEPTH_OFFSETS_H

#include <Texture.h>

#if defined(HAS_CUDA)
#include <CudaGLInterop/CudaGLImage.h>
#endif

namespace quasar {

class DepthOffsets {
public:
    glm::uvec2 size;
    Texture buffer;

    DepthOffsets(const glm::uvec2& size);
    ~DepthOffsets() = default;

#if defined(HAS_CUDA)
    uint copyToMemory(std::vector<char>& outputData);
    uint saveToFile(const std::string& filename);
#endif
    uint loadFromMemory(std::vector<char>& inputData);

private:
    std::vector<char> data;

#if defined(HAS_CUDA)
    CudaGLImage cudaImage;
#endif
};

} // namespace quasar

#endif // DEPTH_OFFSETS_H
