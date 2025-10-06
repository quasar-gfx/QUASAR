#ifndef QUADS_GENERATOR_H
#define QUADS_GENERATOR_H

#include <Cameras/PerspectiveCamera.h>
#include <Shaders/ComputeShader.h>
#include <RenderTargets/FrameRenderTarget.h>

#include <Quads/QuadSet.h>

namespace quasar {

class QuadsGenerator {
public:
    struct BufferSizes {
        uint32_t numProxies;
        uint32_t numProxiesTransparent;
        uint32_t numDepthOffsets;
    };

    struct Stats {
        double generateQuadsTimeMs = 0.0;
        double simplifyQuadsTimeMs = 0.0;
        double gatherQuadsTimeMs = 0.0;
    } stats;

    struct Parameters {
        bool expandEdges = false;
        bool correctOrientation = true;
        float depthThreshold = 1e-4f;
        float angleThreshold = 88.0f;
        float flattenThreshold = 0.2f;
        float proxySimilarityThreshold = 0.5f;
        int maxIterForceMerge = 3;
    } params;

    std::vector<glm::uvec2> quadMapSizes;

    uint numQuadMaps;
    uint maxProxies;

    QuadsGenerator(QuadSet& quadSet);
    ~QuadsGenerator() = default;

    BufferSizes getBufferSizes();
    void createProxiesFromRT(const FrameRenderTarget& frameRT, const PerspectiveCamera& remoteCamera);
    void createProxiesFromTextures(const Texture& colorTexture, const Texture& normalsTexture, const Texture& depthTexture, const PerspectiveCamera& remoteCamera);

private:
    QuadSet& quadSet;

    Buffer sizesBuffer;

    std::vector<QuadBuffers> quadMaps;

    ComputeShader createQuadMapShader;
    ComputeShader simplifyQuadMapShader;
    ComputeShader gatherQuadsShader;

    void generateInitialQuadMap(const Texture& colorTexture, const Texture& normalsTexture, const Texture& depthTexture, const glm::vec2& gBufferSize, const PerspectiveCamera& remoteCamera);
    void simplifyQuadMaps(const PerspectiveCamera& remoteCamera, const glm::vec2& gBufferSize);
    void gatherOutputQuads(const glm::vec2& gBufferSize);
    void createProxies(const Texture& colorTexture, const Texture& normalsTexture, const Texture& depthTexture, const PerspectiveCamera& remoteCamera);
};

} // namespace quasar

#endif // QUADS_GENERATOR_H
