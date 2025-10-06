#ifndef MESH_WARP_STREAMER_H
#define MESH_WARP_STREAMER_H

#include <CameraPose.h>
#include <Cameras/PerspectiveCamera.h>
#include <Renderers/DeferredRenderer.h>
#include <Streamers/VideoStreamer.h>
#include <Streamers/BC4DepthStreamer.h>
#include <PostProcessing/Tonemapper.h>
#include <PostProcessing/ShowDepthEffect.h>

namespace quasar {

class MeshWarpStreamer {
public:
    std::string videoURL;
    std::string depthURL;

    VideoStreamer videoStreamerRT;
    BC4DepthStreamer depthStreamerRT;
    RenderTarget renderTarget;

    MeshWarpStreamer(
        DeferredRenderer& remoteRenderer,
        Scene& remoteScene,
        PerspectiveCamera& remoteCamera,
        const std::string& videoURL = "",
        const std::string& depthURL = "",
        uint depthFactor = 1,
        uint maxFrameRate = 30,
        uint targetBitRate = 12);
    ~MeshWarpStreamer() = default;

    float getVideoFrameRate() { return videoStreamerRT.getFrameRate(); }
    float getDepthFrameRate() { return depthStreamerRT.getFrameRate(); }

    RenderStats generateFrame();
    void sendFrame(pose_id_t poseID);

    size_t writeToFiles(const Path& outputPath);

private:
    DeferredRenderer& remoteRenderer;
    Scene& remoteScene;
    PerspectiveCamera& remoteCamera;

    Tonemapper tonemapper;
    ShowDepthEffect depthEffect;
};

} // namespace quasar

#endif // MESH_WARP_STREAMER_H
