#ifndef QUADS_GENERATOR_H
#define QUADS_GENERATOR_H

#include <Cameras/PerspectiveCamera.h>
#include <Shaders/ComputeShader.h>
#include <RenderTargets/GBuffer.h>

#include <Quads/QuadsBuffers.h>
#include <Quads/DepthOffsets.h>

class QuadsGenerator {
public:
    struct BufferSizes {
        unsigned int numProxies;
        unsigned int numDepthOffsets;
    };

    struct Stats {
        double timeToGenerateQuadsMs = -1.0f;
        double timeToSimplifyQuadsMs = -1.0f;
        double timeToFillOutputQuadsMs = -1.0f;
    } stats;

    bool doOrientationCorrection = true;
    float distanceThreshold = 0.5f;
    float angleThreshold = 87.0f;
    float flatThreshold = 2.0f;
    float proxySimilarityThreshold = 0.2f;

    glm::uvec2 remoteWindowSize;
    glm::uvec2 depthBufferSize;
    std::vector<glm::uvec2> quadMapSizes;

    unsigned int numQuadMaps;
    unsigned int maxQuads;

    QuadBuffers outputQuadBuffers;
    DepthOffsets depthOffsets;

    QuadsGenerator(const glm::uvec2 &remoteWindowSize);
    ~QuadsGenerator() = default;

    BufferSizes getBufferSizes();
    unsigned int createProxiesFromGBuffer(const GeometryBuffer& gBuffer, const PerspectiveCamera &remoteCamera);
#ifdef GL_CORE
    unsigned int saveToFile(const std::string &filename);
    unsigned int saveDepthOffsetsToFile(const std::string &filename);
#endif

private:
    Buffer<BufferSizes> sizesBuffer;

    std::vector<QuadBuffers> quadBuffers;

    ComputeShader genQuadMapShader;
    ComputeShader simplifyQuadMapShader;
    ComputeShader fillOutputQuadsShader;

    void generateInitialQuadMap(const GeometryBuffer& gBuffer, const PerspectiveCamera &remoteCamera);
    void simplifyQuadMaps(const PerspectiveCamera &remoteCamera);
    void fillOutputQuads();
};

#endif // QUADS_GENERATOR_H
