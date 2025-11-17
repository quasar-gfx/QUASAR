#ifndef QUADS_SIMULATOR_H
#define QUADS_SIMULATOR_H

#include <CameraPose.h>
#include <DepthMesh.h>
#include <Quads/FrameGenerator.h>
#include <Receivers/QuadsReceiver.h>
#include <Networking/DataStreamerTCP.h>
#include <Streamers/VideoStreamer.h>
#include <PostProcessing/Tonemapper.h>
#include <PostProcessing/ShowNormalsEffect.h>
#include <Codecs/AlphaCodec.h>

namespace quasar {

struct QuadsStreamerCreateParams {
    uint targetFramerate = 5;
    uint targetBitRate = 16;
    std::string videoURL = "";
    std::string proxiesURL = "";
};

class QuadsStreamer : public DataStreamerTCP {
public:
    // Reference frame
    FrameRenderTarget referenceFrameRT;
    ReferenceFrame referenceFrame;
    std::vector<QuadMesh> referenceFrameMeshes;
    std::vector<Node> referenceFrameNodes;
    int meshIndex = 0, lastMeshIndex = 1;

    // Residual frame
    FrameRenderTarget residualFrameRT;
    // Render target to hold updated/masked depth and normals for residual frames
    FrameRenderTarget residualFrameMaskRT;
    ResidualFrame residualFrame;
    QuadMesh residualFrameMesh;
    Node residualFrameNode;

    VideoStreamer videoAtlasStreamerRT;
    FrameRenderTarget alphaAtlasRT;

    // Local objects
    std::vector<Node> referenceFrameNodesLocal;
    std::vector<Node> referenceFrameWireframesLocal;
    Node residualFrameNodeLocal;
    Node residualFrameWireframeLocal;

    // Depth point cloud for debugging
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

    QuadsStreamer(
        QuadSet& quadSet,
        DeferredRenderer& remoteRenderer,
        Scene& remoteScene,
        PerspectiveCamera& remoteCamera,
        const QuadsStreamerCreateParams& params = {});
    ~QuadsStreamer();

    uint getNumTriangles() const;
    std::shared_ptr<QuadsGenerator> getQuadsGenerator() { return frameGenerator.getQuadsGenerator(); }

    void addMeshesToScene(Scene& localScene);

    RenderStats generateFrame(bool createResidualFrame, bool showNormals = false, bool showDepth = false);
    void sendFrame(pose_id_t poseID, bool createResidualFrame);

    void writeTexturesToFiles(const Path& outputPath);
    size_t writeToFiles(const Path& outputPath);
    size_t writeToMemory(pose_id_t poseID, bool writeResidualFrame, std::vector<char>& outputData);

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

    QuadSet& quadSet;
    FrameGenerator frameGenerator;

    DeferredRenderer& remoteRenderer;
    Scene& remoteScene;

    PerspectiveCamera& remoteCamera;
    PerspectiveCamera remoteCameraPrev;

    // Scenes with resulting meshes
    std::vector<Scene> meshScenes;

    FrameRenderTarget referenceFrameRT_noTone;
    FrameRenderTarget residualFrameRT_noTone;

    std::vector<unsigned char> alphaImageData;

    std::vector<char> cameraData;
    std::vector<char> alphaData;
    std::vector<char> geometryData;
    std::vector<char> compressedData;

    AlphaCodec alphaCodec;

    QuadMaterial wireframeMaterial;
    QuadMaterial maskWireframeMaterial;

    Tonemapper tonemapper;
    ShowNormalsEffect showNormalsEffect;
};

} // namespace quasar

#endif // QUADS_SIMULATOR_H
