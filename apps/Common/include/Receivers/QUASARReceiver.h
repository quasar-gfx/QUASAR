#ifndef QUASAR_RECEIVER_H
#define QUASAR_RECEIVER_H

#include <BS_thread_pool/BS_thread_pool.hpp>

#include <Path.h>
#include <CameraPose.h>
#include <Quads/QuadSet.h>
#include <Quads/QuadFrames.h>
#include <Quads/QuadMesh.h>
#include <Networking/DataReceiverTCP.h>
#include <Receivers/VideoTexture.h>

namespace quasar {

class QUASARReceiver : public DataReceiverTCP {
public:
    struct Stats {
        uint totalTriangles = 0;
        double timeToLoadMs = 0.0;
        double timeToDecompressMs = 0.0;
        double timeToTransferMs = 0.0;
        double timeToCreateMeshMs = 0.0;
        QuadSet::Sizes sizes{};
    } stats;

    std::string proxiesURL;
    std::string videoURL;

    uint maxLayers;
    float viewSphereDiameter;
    std::vector<ReferenceFrame> frames;

    VideoTexture atlasVideoTexture;

    QUASARReceiver(QuadSet& quadSet, uint maxLayers, const std::string& videoURL = "", const std::string& proxiesURL = "");
    QUASARReceiver(QuadSet& quadSet, uint maxLayers, float remoteFOV, float remoteFOVWide, const std::string& videoURL = "", const std::string& proxiesURL = "");
    ~QUASARReceiver() = default;

    QuadMesh& getMesh(int layer) { return meshes[layer]; }
    PerspectiveCamera& getRemoteCamera() { return remoteCamera; }
    PerspectiveCamera& getRemoteCameraPrev() { return remoteCameraPrev; }
    PerspectiveCamera& getremoteCameraWideFOV() { return remoteCameraWideFOV; }
    void copyPoseToCamera(PerspectiveCamera& camera) {
        camera.setViewMatrix(remoteCamera.getViewMatrix());
        camera.setProjectionMatrix(remoteCamera.getProjectionMatrix());
    }

    void updateViewSphere(float viewSphereDiameter);

    void loadFromFiles(const Path& dataPath);

private:
    QuadSet& quadSet;
    PerspectiveCamera remoteCamera;
    PerspectiveCamera remoteCameraWideFOV;
    PerspectiveCamera remoteCameraPrev;

    std::vector<QuadMesh> meshes;

    std::unique_ptr<BS::thread_pool<>> threadPool;

    // Temporary buffers for decompression
    std::vector<char> uncompressedQuads, uncompressedOffsets;

    inline const PerspectiveCamera& getCameraToUse(int layer) const {
        return (layer == maxLayers - 1) ? remoteCameraWideFOV : remoteCamera;
    }

    void onDataReceived(const std::vector<char>& data) override;
};

} // namespace quasar

#endif // QUASAR_RECEIVER_H
