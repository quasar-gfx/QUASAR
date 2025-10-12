#ifndef MESH_WARP_STREAMER_H
#define MESH_WARP_STREAMER_H

#include <CameraPose.h>
#include <Materials/UnlitMaterial.h>
#include <Primitives/Mesh.h>
#include <Renderers/DeferredRenderer.h>
#include <Streamers/VideoStreamer.h>
#include <Streamers/BC4DepthStreamer.h>
#include <PostProcessing/Tonemapper.h>
#include <PostProcessing/ShowDepthEffect.h>
#include <Shaders/ComputeShader.h>

namespace quasar {

struct MeshWarpStreamerCreateParams {
    uint depthFactor = 1;
    uint vertexGroupSize = 1;
    uint maxFrameRate = 30;
    uint targetBitRate = 12;
    std::string videoURL = "";
    std::string depthURL = "";
};

class MeshWarpStreamer {
public:
    std::string videoURL;
    std::string depthURL;

    VideoStreamer videoStreamerRT;
    BC4DepthStreamer depthStreamerRT;
    RenderTarget renderTarget;

    struct Stats {
        double totalRenderTimeMs = 0.0;
        double totalGenMeshTime = 0.0;
        double totalCompressTimeMs = 0.0;
        size_t compressedSize = 0;
    } stats;

    MeshWarpStreamer(
        DeferredRenderer& remoteRenderer,
        Scene& remoteScene,
        PerspectiveCamera& remoteCamera,
        const MeshWarpStreamerCreateParams& params = {});
    ~MeshWarpStreamer() = default;

    float getVideoFrameRate() { return videoStreamerRT.getFrameRate(); }
    float getDepthFrameRate() { return depthStreamerRT.getFrameRate(); }
    Mesh& getMesh() { return mesh; }

    RenderStats generateFrame();
    void sendFrame(pose_id_t poseID);

    size_t writeToFiles(const Path& outputPath);

private:
    glm::uvec2 adjustedSize;
    glm::uvec2 depthMapSize;

    DeferredRenderer& remoteRenderer;
    Scene& remoteScene;
    PerspectiveCamera& remoteCamera;

    Tonemapper tonemapper;
    ShowDepthEffect depthEffect;

    ComputeShader meshFromBC4Shader;

    Mesh mesh;
    UnlitMaterial meshMaterial;
};

} // namespace quasar

#endif // MESH_WARP_STREAMER_H
