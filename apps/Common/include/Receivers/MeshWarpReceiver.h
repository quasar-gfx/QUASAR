#ifndef MESH_WARP_RECEIVER_H
#define MESH_WARP_RECEIVER_H

#include <Path.h>
#include <CameraPose.h>
#include <Materials/UnlitMaterial.h>
#include <Primitives/Mesh.h>
#include <Receivers/VideoTexture.h>
#include <Receivers/BC4DepthVideoTexture.h>
#include <Shaders/Shader.h>
#include <Shaders/ComputeShader.h>
#include <Streamers/PoseStreamer.h>

namespace quasar {

class MeshWarpReceiver {
public:
    std::string videoURL;
    std::string depthURL;

    VideoTexture videoTexture;
    BC4DepthVideoTexture depthTexture;

    bool sync = true;
    Pose colorFramePose, depthFramePose;
    pose_id_t poseIdColor = -1, poseIdDepth = -1;

    MeshWarpReceiver(
        const glm::uvec2& remoteGBufferSize,
        uint depthFactor = 1,
        uint vertexGroupSize = 1,
        float remoteFOV = 140.0f,
        const std::string& videoURL = "",
        const std::string& depthURL = "");
    ~MeshWarpReceiver() = default;

    Mesh& getMesh() { return mesh; }
    PerspectiveCamera& getRemoteCamera() { return remoteCamera; }
    void copyPoseToCamera(PerspectiveCamera& camera) { colorFramePose.copyPoseToCamera(camera); }

    void loadFromFiles(const Path& dataPath);

    void recvData(const PoseStreamer& poseStreamer, double& elapsedTimeColor, double& elapsedTimeDepth);

private:
    uint depthFactor;
    uint vertexGroupSize;
    glm::uvec2 adjustedSize;

    PerspectiveCamera remoteCamera;

    ComputeShader meshFromBC4Shader;

    Mesh mesh;
    UnlitMaterial meshMaterial;

    void updateMesh();
};

} // namespace quasar

#endif // MESH_WARP_RECEIVER_H
