#include <Streamers/MeshWarpStreamer.h>

using namespace quasar;

MeshWarpStreamer::MeshWarpStreamer(
        DeferredRenderer& remoteRenderer,
        Scene& remoteScene,
        PerspectiveCamera& remoteCamera,
        const std::string& videoURL,
        const std::string& depthURL,
        uint depthFactor,
        uint maxFrameRate,
        uint targetBitRate)
    : videoURL(videoURL)
    , depthURL(depthURL)
    , remoteRenderer(remoteRenderer)
    , remoteScene(remoteScene)
    , remoteCamera(remoteCamera)
    , renderTarget({
        .width = remoteRenderer.width,
        .height = remoteRenderer.height,
        .internalFormat = GL_RGBA16F,
        .format = GL_RGBA,
        .type = GL_HALF_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    })
    , videoStreamerRT({
        .width = remoteRenderer.width,
        .height = remoteRenderer.height,
        .internalFormat = GL_SRGB8_ALPHA8,
        .format = GL_RGBA,
        .type = GL_UNSIGNED_BYTE,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR,
    }, videoURL, maxFrameRate, targetBitRate)
    , depthStreamerRT({
        .width = remoteRenderer.width / depthFactor,
        .height = remoteRenderer.height / depthFactor,
        .internalFormat = GL_R32F,
        .format = GL_RED,
        .type = GL_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    }, depthURL)
    , depthEffect(remoteCamera)
{}

RenderStats MeshWarpStreamer::generateFrame() {
    // Render all objects in scene
    RenderStats renderStats = remoteRenderer.drawObjects(remoteScene, remoteCamera);

    // Copy to intermediate render target
    remoteRenderer.outputRT.blit(renderTarget);

    // Copy color and depth to video frames
    tonemapper.drawToRenderTarget(remoteRenderer, videoStreamerRT);
    depthEffect.drawToRenderTarget(remoteRenderer, depthStreamerRT);

    return renderStats;
}

void MeshWarpStreamer::sendFrame(pose_id_t poseID) {
    videoStreamerRT.sendFrame(poseID);
    depthStreamerRT.sendFrame(poseID);
}

size_t MeshWarpStreamer::writeToFiles(const Path& outputPath) {
    double startTime = timeutils::getTimeMicros();

    // Save camera data
    Pose cameraPose;
    Path cameraFileName = (outputPath / "camera").withExtension(".bin");
    cameraPose.setProjectionMatrix(remoteCamera.getProjectionMatrix());
    cameraPose.setViewMatrix(remoteCamera.getViewMatrix());
    cameraPose.writeToFile(cameraFileName);

    // Save color
    Path colorFileName = (outputPath / "color").withExtension(".jpg");
    videoStreamerRT.writeColorAsJPG(colorFileName);

    // Save depth
    Path depthFileName = (outputPath / "depth").withExtension(".bc4.zstd");
    size_t totalBytes = depthStreamerRT.writeToFile(depthFileName);

    spdlog::info("Saved {:.3f}MB in {:.3f}ms",
                 static_cast<double>(totalBytes) / BYTES_PER_MEGABYTE,
                 timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

    return totalBytes;
}
