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
        double timeToGenerateQuadsMs = 0.0f;
        double timeToSimplifyQuadsMs = 0.0f;
        double timeToGatherQuadsMs = 0.0f;
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

    glm::uvec2 &remoteWindowSize;
    glm::uvec2 depthBufferSize;
    std::vector<glm::uvec2> quadMapSizes;

    uint numQuadMaps;
    uint maxProxies;

    QuadBuffers outputQuadBuffers;
    DepthOffsets depthOffsets;

    QuadsGenerator(glm::uvec2 &remoteWindowSize);
    ~QuadsGenerator() = default;

    BufferSizes getBufferSizes();
    void createProxiesFromGBuffer(const FrameRenderTarget& frameRT, const PerspectiveCamera &remoteCamera);
#ifdef GL_CORE
    uint saveQuadsToMemory(std::vector<char> &compressedData, bool compress = true);
    uint saveDepthOffsetsToMemory(std::vector<char> &compressedData, bool compress = true);
    uint saveToFile(const std::string &filename);
    uint saveDepthOffsetsToFile(const std::string &filename);
#endif

private:
    Buffer sizesBuffer;

    std::vector<QuadBuffers> quadBuffers;

    ComputeShader genQuadMapShader;
    ComputeShader simplifyQuadMapShader;
    ComputeShader gatherQuadsShader;

    void generateInitialQuadMap(const FrameRenderTarget& frameRT, const glm::vec2 &gBufferSize, const PerspectiveCamera &remoteCamera);
    void simplifyQuadMaps(const PerspectiveCamera &remoteCamera, const glm::vec2 &gBufferSize);
    void gatherOutputQuads(const glm::vec2 &gBufferSize);
};

} // namespace quasar

#endif // QUADS_GENERATOR_H
