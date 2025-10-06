#ifndef FRAME_GENERATOR_H
#define FRAME_GENERATOR_H

#include <BS_thread_pool/BS_thread_pool.hpp>

#include <Quads/QuadFrames.h>
#include <Quads/QuadMesh.h>
#include <Quads/QuadsGenerator.h>
#include <Renderers/DeferredRenderer.h>
#include <RenderTargets/FrameRenderTarget.h>
#include <Codecs/ZSTDCodec.h>

namespace quasar {

class FrameGenerator {
public:
    struct Stats {
        double createQuadsTimeMs = 0.0;
        double createMeshTimeMs = 0.0;
        double generateQuadsTimeMs = 0.0;
        double simplifyQuadsTimeMs = 0.0;
        double gatherQuadsTimeMs = 0.0;
        double appendQuadsTimeMs = 0.0;
        double createVertIndTimeMs = 0.0;
        double updateRTsTimeMs = 0.0;
        double transferTimeMs = 0.0;
        double compressTimeMs = 0.0;
    } stats;

    struct Params {
        bool applyDeltaEncoding = true;
    } params;

    FrameGenerator(QuadSet& quadSet);

    std::shared_ptr<QuadsGenerator> getQuadsGenerator() { return quadsGenerator; }

    void createReferenceFrame(
        const FrameRenderTarget& referenceFrameRT,
        const PerspectiveCamera& remoteCamera,
        QuadMesh& referenceMesh,
        ReferenceFrame& referenceFrame);

    void updateResidualRenderTargets(
        FrameRenderTarget& residualFrameMaskRT, FrameRenderTarget& residualFrameRT,
        DeferredRenderer& remoteRenderer, Scene& remoteScene,
        Scene& currMeshScene, Scene& prevMeshScene, // Scenes that contain the resulting reconstructed meshes
        const PerspectiveCamera& currRemoteCamera, const PerspectiveCamera& remoteCameraPrev);

    void createResidualFrame(
        const FrameRenderTarget& residualFrameMaskRT, const FrameRenderTarget& residualFrameRT,
        const PerspectiveCamera& currRemoteCamera, const PerspectiveCamera& remoteCameraPrev,
        QuadMesh& referenceMesh, QuadMesh& residualMesh,
        ResidualFrame& residualFrame);

private:
    QuadSet& quadSet;
    std::shared_ptr<QuadsGenerator> quadsGenerator;

    std::unique_ptr<BS::thread_pool<>> threadPool;

    // Temporary buffers for decompression
    std::vector<char> uncompressedQuads, uncompressedOffsets;
    std::vector<char> uncompressedQuadsRevealed, uncompressedOffsetsRevealed;
};

} // namespace quasar

#endif // FRAME_GENERATOR_H
