#ifndef FRAME_GENERATOR_H
#define FRAME_GENERATOR_H

#include <Renderers/DeferredRenderer.h>
#include <RenderTargets/FrameRenderTarget.h>

#include <Quads/QuadsBuffers.h>
#include <Quads/DepthOffsets.h>
#include <Quads/QuadsGenerator.h>
#include <Quads/MeshFromQuads.h>

namespace quasar {

class FrameGenerator {
public:
    QuadsGenerator &quadsGenerator;
    MeshFromQuads &meshFromQuads;

    MeshFromQuads meshFromQuadsMask;

    FrameGenerator(DeferredRenderer &remoteRenderer, const Scene &remoteScene, QuadsGenerator &quadsGenerator, MeshFromQuads &meshFromQuads)
        : remoteRenderer(remoteRenderer)
        , remoteScene(remoteScene)
        , quadsGenerator(quadsGenerator)
        , meshFromQuads(meshFromQuads)
        , meshFromQuadsMask(meshFromQuads.remoteWindowSize, meshFromQuads.maxProxies / 4) {}

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
        const FrameRenderTarget &frameRT,
        const PerspectiveCamera &remoteCamera,
        const Mesh &mesh,
        unsigned int &numProxies, unsigned int &numDepthOffsets,
        bool compress = true);

    unsigned int generatePFrame(
        const Scene &currScene, const Scene &prevScene,
        FrameRenderTarget &frameRT, FrameRenderTarget &frameRTMask,
        const PerspectiveCamera &currRemoteCamera, const PerspectiveCamera &prevRemoteCamera,
        const Mesh &currMesh, const Mesh &maskMesh,
        unsigned int &numProxies, unsigned int &numDepthOffsets,
        bool compress = true);

private:
    std::vector<char> compressedQuads;
    std::vector<char> compressedDepthOffsets;

    DeferredRenderer &remoteRenderer;
    const Scene &remoteScene;
};

} // namespace quasar

#endif // FRAME_GENERATOR_H
