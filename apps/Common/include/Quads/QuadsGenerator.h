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
    float distanceThreshold = 0.001f;
    float angleThreshold = 85.0f;
    float flatThreshold = 0.1f;
    float proxySimilarityThreshold = 1.0f;

    glm::uvec2 remoteWindowSize;
    glm::uvec2 depthBufferSize;
    std::vector<glm::uvec2> quadMapSizes;

    unsigned int numQuadMaps;
    unsigned int maxProxies;

    QuadBuffers outputQuadBuffers;
    DepthOffsets depthOffsets;

    QuadsGenerator(const glm::uvec2 &remoteWindowSize);
    ~QuadsGenerator() = default;

    BufferSizes getBufferSizes();
    BufferSizes createProxiesFromGBuffer(const GBuffer& gBuffer, const PerspectiveCamera &remoteCamera);
#ifdef GL_CORE
    unsigned int saveQuadsToMemory(std::vector<char> &compressedData);
    unsigned int saveDepthOffsetsToMemory(std::vector<char> &compressedData);
    unsigned int saveToFile(const std::string &filename);
    unsigned int saveDepthOffsetsToFile(const std::string &filename);
#endif

private:
    Buffer<BufferSizes> sizesBuffer;

    std::vector<QuadBuffers> quadBuffers;

    ComputeShader genQuadMapShader;
    ComputeShader simplifyQuadMapShader;
    ComputeShader fillOutputQuadsShader;

    void generateInitialQuadMap(const GBuffer& gBuffer, const glm::vec2 &gBufferSize, const PerspectiveCamera &remoteCamera);
    void simplifyQuadMaps(const PerspectiveCamera &remoteCamera, const glm::vec2 &gBufferSize);
    void fillOutputQuads(const glm::vec2 &gBufferSize);
};

#endif // QUADS_GENERATOR_H
