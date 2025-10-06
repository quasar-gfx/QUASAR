#ifndef QUADSTREAM_SIMULATOR_H
#define QUADSTREAM_SIMULATOR_H

#include <CameraPose.h>
#include <DepthMesh.h>
#include <Quads/FrameGenerator.h>
#include <Networking/DataStreamerTCP.h>
#include <Streamers/VideoStreamer.h>
#include <PostProcessing/Tonemapper.h>

#include <UI/FrameRateWindow.h>
#include <PostProcessing/ShowNormalsEffect.h>

namespace quasar {

struct QuadStreamStreamerCreateParams {
    uint maxViews = 4;
    float viewBoxSize = 0.5f;
    float wideFOV = 120.0f;
};

class QuadStreamStreamer {
public:
    uint maxViews;
    float viewBoxSize;

    // Reference frame -- QS has no temporal compression (i.e. no residual frames)
    std::vector<FrameRenderTarget> referenceFrameRTs;
    std::vector<ReferenceFrame> referenceFrames;
    std::vector<QuadMesh> referenceFrameMeshes;
    std::vector<Node> referenceFrameNodesRemote;

    // Local objects
    std::vector<Node> referenceFrameNodesLocal;
    std::vector<Node> referenceFrameWireframesLocal;

    // Depth point cloud for debugging
    std::vector<DepthMesh> depthMeshes;
    std::vector<Node> depthNodes;

    struct Stats {
        double totalRenderTimeMs = 0.0;
        double totalCreateProxiesTimeMs = 0.0;
        double totalGenQuadMapTimeMs = 0.0;
        double totalSimplifyTimeMs = 0.0;
        double totalGatherQuadsTime = 0.0;
        double totalCreateMeshTimeMs = 0.0;
        double totalAppendQuadsTimeMs = 0.0;
        double totalCreateVertIndTimeMs = 0.0;
        double totalGenDepthTimeMs = 0.0;
        double totalCompressTimeMs = 0.0;
        double frameSize = 0.0;
        QuadSet::Sizes proxySizes;
    } stats;

    QuadStreamStreamer(
        QuadSet& quadSet,
        DeferredRenderer& remoteRenderer,
        Scene& remoteScene,
        PerspectiveCamera& remoteCamera,
        const QuadStreamStreamerCreateParams& params = {});
    ~QuadStreamStreamer() = default;

    uint getNumTriangles() const;
    std::shared_ptr<QuadsGenerator> getQuadsGenerator() { return frameGenerator.getQuadsGenerator(); }

    void addMeshesToScene(Scene& localScene);
    void setViewBoxSize(float viewBoxSize);

    RenderStats generateFrame(bool showNormals = false, bool showDepth = false);

    size_t writeToFiles(const Path& outputPath);

private:
    const std::vector<glm::vec4> colors = {
        glm::vec4(1.0f, 1.0f, 0.0f, 1.0f), // primary view color is yellow
        glm::vec4(0.0f, 0.0f, 1.0f, 1.0f),
        glm::vec4(0.0f, 1.0f, 0.0f, 1.0f),
        glm::vec4(1.0f, 0.5f, 0.5f, 1.0f),
        glm::vec4(0.5f, 0.0f, 0.0f, 1.0f),
        glm::vec4(0.0f, 1.0f, 1.0f, 1.0f),
        glm::vec4(1.0f, 0.0f, 0.0f, 1.0f),
        glm::vec4(0.0f, 0.5f, 0.0f, 1.0f),
        glm::vec4(0.0f, 0.0f, 0.5f, 1.0f),
        glm::vec4(0.5f, 0.0f, 0.5f, 1.0f),
    };

    const std::vector<glm::vec3> offsets = {
        glm::vec3(-1.0f, +1.0f, -1.0f), // Top-left
        glm::vec3(+1.0f, +1.0f, -1.0f), // Top-right
        glm::vec3(+1.0f, -1.0f, -1.0f), // Bottom-right
        glm::vec3(-1.0f, -1.0f, -1.0f), // Bottom-left
        glm::vec3(-1.0f, +1.0f, +1.0f), // Top-left
        glm::vec3(+1.0f, +1.0f, +1.0f), // Top-right
        glm::vec3(+1.0f, -1.0f, +1.0f), // Bottom-right
        glm::vec3(-1.0f, -1.0f, +1.0f), // Bottom-left
    };

    QuadSet& quadSet;
    FrameGenerator frameGenerator;

    DeferredRenderer& remoteRenderer;
    Scene& remoteScene;
    PerspectiveCamera& remoteCamera;
    std::vector<PerspectiveCamera> remoteCameras;

    // Scenes with resulting meshes
    Scene meshScene;

    // Holds a copies of the current frame
    std::vector<FrameRenderTarget> referenceFrameRTs_noTone;

    // Shaders
    Tonemapper tonemapper;
    ShowNormalsEffect showNormalsEffect;

    QuadMaterial wireframeMaterial;
    QuadMaterial maskWireframeMaterial;
};

} // namespace quasar

#endif // QUADSTREAM_SIMULATOR_H
