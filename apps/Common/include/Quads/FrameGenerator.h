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
    QuadsGenerator& quadsGenerator;
    MeshFromQuads& meshFromQuads;

    MeshFromQuads meshFromQuadsMask;

    FrameGenerator(DeferredRenderer& remoteRenderer, Scene& remoteScene, QuadsGenerator& quadsGenerator, MeshFromQuads& meshFromQuads);

    struct Stats {
        double timeToCreateProxiesMs = 0.0;
        double timeToCreateMeshMs = 0.0;
        double timeToGenerateQuadsMs = 0.0;
        double timeToSimplifyQuadsMs = 0.0;
        double timeToGatherQuadsMs = 0.0;
        double timeToAppendQuadsMs = 0.0;
        double timeToFillQuadIndicesMs = 0.0;
        double timeToCreateVertIndMs = 0.0;
        double timeToRenderMaskMs = 0.0;
        double timeToCompress = 0.0f;
    } stats;

    uint generateRefFrame(
        const FrameRenderTarget& frameRT,
        const PerspectiveCamera& remoteCamera,
        const Mesh& mesh,
        std::vector<char>& quads, std::vector<char>& depthOffsets,
        uint& numProxies, uint& numDepthOffsets,
        bool compress = true);

    uint generateResFrame(
        Scene& currScene, Scene& prevScene,
        FrameRenderTarget& frameRT, FrameRenderTarget& maskFrameRT,
        const PerspectiveCamera& currRemoteCamera, const PerspectiveCamera& prevRemoteCamera,
        const Mesh& currMesh, const Mesh& maskMesh,
        std::vector<char>& quads, std::vector<char>& depthOffsets,
        uint& numProxies, uint& numDepthOffsets,
        bool compress = true);

private:
    DeferredRenderer& remoteRenderer;
    Scene& remoteScene;
};

} // namespace quasar

#endif // FRAME_GENERATOR_H
