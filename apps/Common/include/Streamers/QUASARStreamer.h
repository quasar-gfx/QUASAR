#ifndef QUASAR_SIMULATOR_H
#define QUASAR_SIMULATOR_H

#include <CameraPose.h>
#include <DepthMesh.h>
#include <Quads/FrameGenerator.h>
#include <Receivers/QUASARReceiver.h>
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
    int meshIndex = 0, lastMeshIndex = 1;

    // Residual frame -- we only create the residuals to the visible layer
    FrameRenderTarget residualFrameRT;
    // Render target to hold updated/masked depth and normals for residual frames
    FrameRenderTarget residualFrameMaskRT;
    ResidualFrame residualFrame;
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
        DepthPeelingRenderer& remoteRendererDP,
        DeferredRenderer& remoteRenderer,
        Scene& remoteScene,
        PerspectiveCamera& remoteCamera,
        float viewSphereDiameter,
        float wideFOV,
        const std::string& videoURL = "",
        const std::string& proxiesURL = "",
        uint targetBitRate = 20);
    ~QUASARStreamer();

    uint getNumTriangles() const;
    std::shared_ptr<QuadsGenerator> getQuadsGenerator() { return frameGenerator.getQuadsGenerator(); }

    void addMeshesToScene(Scene& localScene);
    void setViewSphereDiameter(float viewSphereDiameter);

    void generateFrame(bool createResidualFrame = false, bool showNormals = false, bool showDepth = false);
    void sendProxies(pose_id_t poseID, bool createResidualFrame);

    size_t writeToFiles(const Path& outputPath);
    size_t writeToMemory(pose_id_t poseID, bool writeResidualFrame, std::vector<char>& outputData);

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

    DepthPeelingRenderer& remoteRendererDP;
    DeferredRenderer& remoteRenderer;
    Scene& remoteScene;
    PerspectiveCamera& remoteCamera;
    PerspectiveCamera remoteCameraPrev;
    PerspectiveCamera remoteCameraWideFOV;

    // Scenes with resulting meshes
    std::vector<Scene> meshScenes;
    Scene sceneWideFov;

    FrameRenderTarget referenceFrameRT_noTone;
    FrameRenderTarget residualFrameRT_noTone;
    std::vector<FrameRenderTarget> frameRTsHidLayer_noTone;

    std::vector<char> cameraData;
    std::vector<std::vector<char>> geometryMetadatas;
    std::vector<char> compressedData;

    QuadMaterial wireframeMaterial;
    QuadMaterial maskWireframeMaterial;

    ToneMapper toneMapper;
    ShowNormalsEffect showNormalsEffect;
};

} // namespace quasar

#endif // QUASAR_SIMULATOR_H
