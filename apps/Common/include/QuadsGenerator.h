#ifndef QUADS_GENERATOR_H
#define QUADS_GENERATOR_H

#include <Cameras/PerspectiveCamera.h>
#include <Shaders/ComputeShader.h>
#include <RenderTargets/GBuffer.h>

#if !defined(__APPLE__) && !defined(__ANDROID__)
#include <cuda_gl_interop.h>
#include <Utils/CudaUtils.h>
#endif

struct QuadMapDataPacked {
    // normal converted into spherical coordinates. theta, phi (16 bits each) packed into 32 bits
    unsigned int normalSpherical;
    // full resolution depth (32 bits)
    float depth;
    // x << 12 | y (12 bits each). 24 bits used
    unsigned int xy;
    // offset.x << 20 | offset.y << 8 (12 bits each) | size << 1 (5 bits) | flattened (1 bit). 30 bits used
    unsigned int offsetSizeFlattened;
}; // 128 bits total

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

    glm::uvec2 remoteWindowSize;
    glm::uvec2 depthBufferSize;

    unsigned int numQuadMaps;
    unsigned int maxQuads;

    bool doOrientationCorrection = true;
    float distanceThreshold = 0.5f;
    float angleThreshold = 87.0f;
    float flatThreshold = 2.0f;
    float proxySimilarityThreshold = 0.2f;

    std::vector<glm::uvec2> quadMapSizes;

    Buffer<unsigned int> outputNormalSphericalsBuffer;
    Buffer<float> outputDepthsBuffer;
    Buffer<unsigned int> outputXYsBuffer;
    Buffer<unsigned int> outputOffsetSizeFlattenedsBuffer;

    Texture depthOffsetsBuffer;

    QuadsGenerator(const glm::uvec2 &remoteWindowSize);
    ~QuadsGenerator();

    void generateInitialQuadMap(const GeometryBuffer& gBuffer, const PerspectiveCamera &remoteCamera);
    void simplifyQuadMaps(const PerspectiveCamera &remoteCamera);
    void fillOutputQuads();
    void createProxiesFromGBuffer(const GeometryBuffer& gBuffer, const PerspectiveCamera &remoteCamera);

    BufferSizes getBufferSizes();

#ifdef GL_CORE
    unsigned int getProxies(char* proxiesData);
    unsigned int saveProxies(const std::string &filename);
#endif

private:
    Buffer<BufferSizes> sizesBuffer;
    char* proxiesData = nullptr;

    std::vector<Buffer<unsigned int>> normalSphericalsBuffers;
    std::vector<Buffer<float>> depthsBuffers;
    std::vector<Buffer<unsigned int>> xysBuffers;
    std::vector<Buffer<unsigned int>> offsetSizeFlattenedsBuffers;

    ComputeShader genQuadMapShader;
    ComputeShader simplifyQuadMapShader;
    ComputeShader fillOutputQuadsShader;

#if !defined(__APPLE__) && !defined(__ANDROID__)
    cudaGraphicsResource* cudaResourceNormalSphericals;
    cudaGraphicsResource* cudaResourceDepths;
    cudaGraphicsResource* cudaResourceXys;
    cudaGraphicsResource* cudaResourceOffsetSizeFlatteneds;
    cudaGraphicsResource* cudaResource;
#endif

    void initializeBuffers();
};

#endif // QUADS_GENERATOR_H
