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
        double timeToGatherQuadsMs = 0.0f;
        double timeToAppendQuadsMs = 0.0f;
        double timeToFillQuadIndicesMs = 0.0f;
        double timeToCreateVertIndMs = 0.0f;
        double timeToRenderMaskMs = 0.0f;
        double timeToCompress = 0.0f;
    } stats;

    uint generateRefFrame(
        const FrameRenderTarget &frameRT,
        const PerspectiveCamera &remoteCamera,
        const Mesh &mesh,
        std::vector<char> &quads, std::vector<char> &depthOffsets,
        uint &numProxies, uint &numDepthOffsets,
        bool compress = true);

    uint generateResFrame(
        const Scene &currScene, const Scene &prevScene,
        FrameRenderTarget &frameRT, FrameRenderTarget &maskFrameRT,
        const PerspectiveCamera &currRemoteCamera, const PerspectiveCamera &prevRemoteCamera,
        const Mesh &currMesh, const Mesh &maskMesh,
        std::vector<char> &quads, std::vector<char> &depthOffsets,
        uint &numProxies, uint &numDepthOffsets,
        bool compress = true);

private:
    DeferredRenderer &remoteRenderer;
    const Scene &remoteScene;
};

} // namespace quasar

#endif // FRAME_GENERATOR_H
