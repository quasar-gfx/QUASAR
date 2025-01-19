#ifndef FRAME_GENERATOR_H
#define FRAME_GENERATOR_H

#include <Renderers/ForwardRenderer.h>
#include <RenderTargets/GBuffer.h>

#include <Quads/QuadsBuffers.h>
#include <Quads/DepthOffsets.h>
#include <Quads/QuadsGenerator.h>
#include <Quads/MeshFromQuads.h>

class FrameGenerator {
public:
    FrameGenerator() = default;

    struct Stats {
        double timeToCreateProxies = 0.0f;
        double timeToCreateMesh = 0.0f;
        double timeToGenerateQuads = 0.0f;
        double timeToSimplifyQuads = 0.0f;
        double timeToFillOutputQuads = 0.0f;
        double timeToAppendProxies = 0.0f;
        double timeToFillQuadIndices = 0.0f;
        double timeToCreateVertInd = 0.0f;
        double timeToRenderMasks = 0.0f;
    } stats;

    unsigned int generateIFrame(
        const GBuffer &gBuffer, const PerspectiveCamera &remoteCamera,
        QuadsGenerator &quadsGenerator, MeshFromQuads &meshFromQuads, const Mesh &mesh,
        unsigned int &numProxies, unsigned int &numDepthOffsets);

    unsigned int generatePFrame(
        ForwardRenderer &remoteRenderer, const Scene &remoteScene, const Scene &currScene, const Scene &prevScene,
        GBuffer &gBuffer, GBuffer &gBufferMask,
        const PerspectiveCamera &currRemoteCamera, const PerspectiveCamera &prevRemoteCamera,
        QuadsGenerator &quadsGenerator, MeshFromQuads &meshFromQuads, MeshFromQuads &meshFromQuadsMask,
        const Mesh &currMesh, const Mesh &maskMesh,
        unsigned int &numProxies, unsigned int &numDepthOffsets);

private:
    std::vector<char> compressedQuads;
    std::vector<char> compressedDepthOffsets;
};

#endif // FRAME_GENERATOR_H
