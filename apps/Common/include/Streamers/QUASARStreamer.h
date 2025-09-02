#ifndef QUASAR_SIMULATOR_H
#define QUASAR_SIMULATOR_H

#include <CameraPose.h>
#include <DepthMesh.h>
#include <Quads/FrameGenerator.h>
#include <Renderers/DepthPeelingRenderer.h>
#include <Networking/DataStreamerTCP.h>
#include <Streamers/VideoStreamer.h>
#include <PostProcessing/ToneMapper.h>
#include <PostProcessing/ShowNormalsEffect.h>

namespace quasar {

class QUASARStreamer : public DataStreamerTCP {
public:
    uint maxLayers;
    float viewSphereDiameter;

    // Reference frame
    FrameRenderTarget referenceFrameRT;
    std::vector<ReferenceFrame> referenceFrames;
    std::vector<QuadMesh> referenceFrameMeshes;
    std::vector<Node> referenceFrameNodes;
    std::vector<Node> referenceFrameWireframesLocal;
    int currMeshIndex = 0, prevMeshIndex = 1;

    // Residual frame -- we only create the residuals to the visible layer
    FrameRenderTarget residualFrameRT;
    // Render target to hold updated/masked depth and normals for residual frames
    FrameRenderTarget residualFrameMaskRT;
    std::vector<ResidualFrame> residualFrames;
    QuadMesh residualFrameMesh;
    Node residualFrameNode;

    VideoStreamer atlasVideoStreamerRT;

    // Local objects
    std::vector<Node> referenceFrameNodesLocal;
    Node residualFrameWireframesLocal;

    // Hidden layers
    std::vector<FrameRenderTarget> frameRTsHidLayer;
    std::vector<QuadMesh> meshesHidLayer;
    std::vector<Node> nodesHidLayer;
    std::vector<Node> wireframesHidLayer;

    // Depth point cloud for debugging
    std::vector<DepthMesh> depthMeshsHidLayer;
    std::vector<Node> depthNodesHidLayer;

    // Wide fov
    std::vector<Node> wideFovNodes;

    DepthMesh depthMesh;
    Node depthNode;

    std::string videoURL;
    std::string proxiesURL;

    struct Stats {
        double totalRenderTime = 0.0;
        double totalCreateProxiesTime = 0.0;
        double totalGenQuadMapTime = 0.0;
        double totalSimplifyTime = 0.0;
        double totalGatherQuadsTime = 0.0;
        double totaltimeToCreateMeshMs = 0.0;
        double totalAppendQuadsTime = 0.0;
        double totalFillQuadsIndiciesTime = 0.0;
        double totalCreateVertIndTime = 0.0;
        double totalGenDepthTime = 0.0;
        double totalCompressTime = 0.0;
        QuadSet::Sizes totalSizes;
    } stats;

    QUASARStreamer(
        QuadSet& quadSet,
        uint maxLayers,
        DeferredRenderer& remoteRenderer,
        Scene& remoteScene,
        const PerspectiveCamera& remoteCamera,
        float wideFOV,
        const std::string& videoURL = "",
        const std::string& proxiesURL = "");
    ~QUASARStreamer() = default;

    uint getNumTriangles() const;
    std::shared_ptr<QuadsGenerator> getQuadsGenerator() { return frameGenerator.getQuadsGenerator(); }

    void addMeshesToScene(Scene& localScene);

    void updateViewSphere(const PerspectiveCamera& remoteCamera, float viewSphereDiameter);
    void generateFrame(
        DeferredRenderer& remoteRenderer, DepthPeelingRenderer& remoteRendererDP,
        Scene& remoteScene, const PerspectiveCamera& remoteCamera,
        bool createResidualFrame = false, bool showNormals = false, bool showDepth = false);

    size_t writeToFiles(const PerspectiveCamera& remoteCamera, const Path& outputPath);

private:
    const std::vector<glm::vec4> colors = {
        glm::vec4(1.0f, 1.0f, 0.0f, 1.0f), // primary layer color is yellow
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

    QuadSet& quadSet;
    FrameGenerator frameGenerator;

    DeferredRenderer& remoteRenderer;
    Scene& remoteScene;

    PerspectiveCamera remoteCameraPrev;
    PerspectiveCamera remoteCameraWideFOV;

    // Scenes with resulting meshes
    std::vector<Scene> meshScenes;
    Scene sceneWideFov;

    FrameRenderTarget referenceFrameRT_noTone;
    FrameRenderTarget residualFrameRT_noTone;
    std::vector<FrameRenderTarget> frameRTsHidLayer_noTone;

    QuadMaterial wireframeMaterial;
    QuadMaterial maskWireframeMaterial;

    ToneMapper toneMapper;
    ShowNormalsEffect showNormalsEffect;
};

} // namespace quasar

#endif // QUASAR_SIMULATOR_H
