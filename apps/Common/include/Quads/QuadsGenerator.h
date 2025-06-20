#ifndef QUADS_GENERATOR_H
#define QUADS_GENERATOR_H

#include <Cameras/PerspectiveCamera.h>
#include <Shaders/ComputeShader.h>
#include <RenderTargets/FrameRenderTarget.h>

#include <Quads/QuadsBuffers.h>
#include <Quads/DepthOffsets.h>

namespace quasar {

class QuadsGenerator {
public:
    struct BufferSizes {
        uint numProxies;
        uint numDepthOffsets;
    };

    struct Stats {
        double timeToGenerateQuadsMs = 0.0;
        double timeToSimplifyQuadsMs = 0.0;
        double timeToGatherQuadsMs = 0.0;
    } stats;

    struct Parameters {
        bool expandEdges = false;
        bool correctOrientation = true;
        float depthThreshold = 1e-4f;
        float angleThreshold = 87.5f;
        float flattenThreshold = 0.2f;
        float proxySimilarityThreshold = 0.5f;
        int maxIterForceMerge = 3;
    } params;

    glm::uvec2& remoteWindowSize;
    glm::uvec2 depthOffsetBufferSize;
    std::vector<glm::uvec2> quadMapSizes;

    uint numQuadMaps;
    uint maxProxies;

    QuadBuffers quadBuffers;
    DepthOffsets depthOffsets;

    QuadsGenerator(glm::uvec2& remoteWindowSize);
    ~QuadsGenerator() = default;

    BufferSizes getBufferSizes();
    void createProxiesFromRT(const FrameRenderTarget& frameRT, const PerspectiveCamera& remoteCamera);
    void createProxiesFromTextures(const Texture& colorBuffer, const Texture& normalsBuffer, const Texture& depthBuffer, const PerspectiveCamera& remoteCamera);
#ifdef GL_CORE
    uint saveQuadsToMemory(std::vector<char>& compressedData, bool compress = true);
    uint saveDepthOffsetsToMemory(std::vector<char>& compressedData, bool compress = true);
    uint saveToFile(const std::string& filename);
    uint saveDepthOffsetsToFile(const std::string& filename);
#endif

private:
    Buffer sizesBuffer;

    std::vector<QuadBuffers> quadMaps;

    ComputeShader genQuadMapShader;
    ComputeShader simplifyQuadMapShader;
    ComputeShader gatherQuadsShader;

    void generateInitialQuadMap(const Texture& colorBuffer, const Texture& normalsBuffer, const Texture& depthBuffer, const glm::vec2& gBufferSize, const PerspectiveCamera& remoteCamera);
    void simplifyQuadMaps(const PerspectiveCamera& remoteCamera, const glm::vec2& gBufferSize);
    void gatherOutputQuads(const glm::vec2& gBufferSize);
    void createProxies(const Texture& colorBuffer, const Texture& normalsBuffer, const Texture& depthBuffer, const PerspectiveCamera& remoteCamera);
};

} // namespace quasar

#endif // QUADS_GENERATOR_H
