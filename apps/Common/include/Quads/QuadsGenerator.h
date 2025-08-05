#ifndef QUADS_GENERATOR_H
#define QUADS_GENERATOR_H

#include <Cameras/PerspectiveCamera.h>
#include <Shaders/ComputeShader.h>
#include <RenderTargets/FrameRenderTarget.h>

#include <Quads/QuadFrame.h>

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

    std::vector<glm::uvec2> quadMapSizes;

    uint numQuadMaps;
    uint maxProxies;

    QuadsGenerator(QuadFrame& quadFrame);
    ~QuadsGenerator() = default;

    BufferSizes getBufferSizes();
    void createProxiesFromRT(const FrameRenderTarget& frameRT, const PerspectiveCamera& remoteCamera);
    void createProxiesFromTextures(const Texture& colorBuffer, const Texture& normalsBuffer, const Texture& depthBuffer, const PerspectiveCamera& remoteCamera);

private:
    QuadFrame& quadFrame;

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
