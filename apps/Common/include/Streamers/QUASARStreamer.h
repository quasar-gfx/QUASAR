#ifndef QUASAR_SIMULATOR_H
#define QUASAR_SIMULATOR_H

#include <CameraPose.h>
#include <DepthMesh.h>
#include <Quads/FrameGenerator.h>
#include <Receivers/QUASARReceiver.h>
#include <Renderers/DepthPeelingRenderer.h>
#include <Networking/DataStreamerTCP.h>
#include <Streamers/VideoStreamer.h>
#include <PostProcessing/Tonemapper.h>

#include <UI/FrameRateWindow.h>
#include <PostProcessing/ShowNormalsEffect.h>
#include <Codecs/AlphaCodec.h>

namespace quasar {

struct QUASARStreamerCreateParams {
    uint maxLayers = 5;
    float viewSphereDiameter = 1.0f;
    float wideFOV = 120.0f;
    uint targetFramerate = 5;
    uint targetBitRate = 28;
    std::string videoURL = "";
    std::string proxiesURL = "";
};

class QUASARStreamer : public DataStreamerTCP {
public:
    uint maxLayers;
    float viewSphereDiameter;

    // Reference frame
    FrameRenderTarget referenceFrameRT;
    std::vector<ReferenceFrame> referenceFrames;
    std::vector<QuadMesh> referenceFrameMeshes;
    std::vector<Node> referenceFrameNodes;
    int meshIndex = 0, lastMeshIndex = 1;

    // Residual frame -- we only create the residuals to the visible layer
    FrameRenderTarget residualFrameRT;
    // Render target to hold updated/masked depth and normals for residual frames
    FrameRenderTarget residualFrameMaskRT;
    ResidualFrame residualFrame;
    QuadMesh residualFrameMesh;
    Node residualFrameNode;

    VideoStreamer videoAtlasStreamerRT;
    FrameRenderTarget alphaAtlasRT;

    // Local objects
    std::vector<Node> referenceFrameWireframesLocal;
    std::vector<Node> referenceFrameNodesLocal;
    Node residualFrameNodeLocal;
    Node residualFrameWireframeLocal;

    // Hidden layers
    std::vector<FrameRenderTarget> frameRTsHidLayer;
    std::vector<QuadMesh> meshesHidLayer;
    std::vector<Node> nodesHidLayer;
    std::vector<Node> wireframesHidLayer;

    // Depth point cloud for debugging
    std::vector<DepthMesh> depthMeshesHidLayer;
    std::vector<Node> depthNodesHidLayer;

    // Wide fov
    std::vector<Node> wideFovNodes;

    DepthMesh depthMesh;
    Node depthNode;

    std::string videoURL;
    std::string proxiesURL;

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

    QUASARStreamer(
        QuadSet& quadSet,
        DepthPeelingRenderer& remoteRendererDP,
        DeferredRenderer& remoteRenderer,
        Scene& remoteScene,
        PerspectiveCamera& remoteCamera,
        const QUASARStreamerCreateParams& params = {});
    ~QUASARStreamer();

    uint getNumTriangles() const;
    std::shared_ptr<QuadsGenerator> getQuadsGenerator() { return frameGenerator.getQuadsGenerator(); }

    void addMeshesToScene(Scene& localScene);
    void setViewSphereDiameter(float viewSphereDiameter);

    RenderStats generateFrame(bool createResidualFrame = false, bool showNormals = false, bool showDepth = false);
    void sendFrame(pose_id_t poseID, bool createResidualFrame);

    void setDrawState(QuadMesh::DrawState drawState);

    void writeTexturesToFiles(const Path& outputPath);
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

    std::vector<unsigned char> alphaImageData;

    std::vector<char> cameraData;
    std::vector<char> alphaData;
    std::vector<std::vector<char>> geometryMetadatas;
    std::vector<char> compressedData;

    AlphaCodec alphaCodec;

    QuadMaterial wireframeMaterial;
    QuadMaterial maskWireframeMaterial;

    Tonemapper tonemapper;
    ShowNormalsEffect showNormalsEffect;
};

} // namespace quasar

#endif // QUASAR_SIMULATOR_H
