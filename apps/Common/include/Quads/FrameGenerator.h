#ifndef FRAME_GENERATOR_H
#define FRAME_GENERATOR_H

#include <future>

#include <Quads/QuadFrames.h>
#include <Quads/QuadMesh.h>
#include <Quads/QuadsGenerator.h>
#include <Renderers/DeferredRenderer.h>
#include <RenderTargets/FrameRenderTarget.h>

#include <Codec/ZSTDCodec.h>

namespace quasar {

class FrameGenerator {
public:
    struct Stats {
        double timeToCreateQuadsMs = 0.0;
        double timeToCreateMeshMs = 0.0;
        double timeToGenerateQuadsMs = 0.0;
        double timeToSimplifyQuadsMs = 0.0;
        double timeToGatherQuadsMs = 0.0;
        double timeToAppendQuadsMs = 0.0;
        double timeToFillQuadIndicesMs = 0.0;
        double timeToCreateVertIndMs = 0.0;
        double timeToRenderMaskMs = 0.0;
        double timeToTransferMs = 0.0;
        double timeToCompressMs = 0.0;
    } stats;

    QuadsGenerator quadsGenerator;

    FrameGenerator(QuadSet& quadSet, DeferredRenderer& remoteRenderer, Scene& remoteScene);

    void generateRefFrame(
        const FrameRenderTarget& frameRT,
        const PerspectiveCamera& remoteCamera,
        QuadMesh& mesh,
        ReferenceFrame& frame);

    void generateResFrame(
        Scene& currScene, Scene& prevScene,
        FrameRenderTarget& resFrameRT,
        const PerspectiveCamera& currRemoteCamera, const PerspectiveCamera& prevRemoteCamera,
        QuadMesh& currMesh, QuadMesh& maskMesh,
        ResidualFrame& resultFrame);

private:
    QuadSet& quadSet;

    DeferredRenderer& remoteRenderer;
    Scene& remoteScene;

    // Temporary render target to hold updated/masked depth and normals for residual frames
    FrameRenderTarget maskRT;

    ZSTDCodec refQuadsCodec;
    ZSTDCodec refDepthOffsetsCodec;

    ZSTDCodec resQuadsUpdatedCodec;
    ZSTDCodec resDepthOffsetsUpdatedCodec;
    ZSTDCodec resQuadsRevealedCodec;
    ZSTDCodec resDepthOffsetsRevealedCodec;

    // Temporary buffers for decompression
    std::vector<char> uncompressedQuads, uncompressedOffsets;
    std::vector<char> uncompressedQuadsRevealed, uncompressedOffsetsRevealed;
    std::vector<char> uncompressedQuadsUpdated, uncompressedOffsetsUpdated;
};

} // namespace quasar

#endif // FRAME_GENERATOR_H
