#ifndef FRAME_GENERATOR_H
#define FRAME_GENERATOR_H

#include <Renderers/DeferredRenderer.h>
#include <RenderTargets/GBuffer.h>

#include <Quads/QuadsBuffers.h>
#include <Quads/DepthOffsets.h>
#include <Quads/QuadsGenerator.h>
#include <Quads/MeshFromQuads.h>

class FrameGenerator {
public:
    FrameGenerator() = default;

    struct Stats {
        double timeToCreateProxiesMs = 0.0f;
        double timeToCreateMeshMs = 0.0f;
        double timeToGenerateQuadsMs = 0.0f;
        double timeToSimplifyQuadsMs = 0.0f;
        double timeToFillOutputQuadsMs = 0.0f;
        double timeToAppendProxiesMs = 0.0f;
        double timeToFillQuadIndicesMs = 0.0f;
        double timeToCreateVertIndMs = 0.0f;
        double timeToRenderMasks = 0.0f;
        double timeToCompress = 0.0f;
    } stats;

    unsigned int generateIFrame(
        const GBuffer &gBuffer, const GBuffer &gBufferHighRes,
        const PerspectiveCamera &remoteCamera,
        QuadsGenerator &quadsGenerator, MeshFromQuads &meshFromQuads, const Mesh &mesh,
        unsigned int &numProxies, unsigned int &numDepthOffsets,
        bool compress = true);

    unsigned int generatePFrame(
        DeferredRenderer &remoteRenderer, const Scene &remoteScene, const Scene &currScene, const Scene &prevScene,
        GBuffer &gBufferHighRes, GBuffer &gBufferMaskHighRes,
        GBuffer &gBufferLowRes, GBuffer &gBufferMaskLowRes,
        const PerspectiveCamera &currRemoteCamera, const PerspectiveCamera &prevRemoteCamera,
        QuadsGenerator &quadsGenerator, MeshFromQuads &meshFromQuads, MeshFromQuads &meshFromQuadsMask,
        const Mesh &currMesh, const Mesh &maskMesh,
        unsigned int &numProxies, unsigned int &numDepthOffsets,
        bool compress = true);

private:
    std::vector<char> compressedQuads;
    std::vector<char> compressedDepthOffsets;
};

#endif // FRAME_GENERATOR_H
