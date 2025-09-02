#ifndef QUADSTREAM_RECEIVER_H
#define QUADSTREAM_RECEIVER_H

#include <BS_thread_pool/BS_thread_pool.hpp>

#include <Path.h>
#include <Quads/QuadSet.h>
#include <Quads/QuadFrames.h>
#include <Quads/QuadMesh.h>
#include <Quads/QuadMaterial.h>
#include <CameraPose.h>

namespace quasar {

class QuadStreamReceiver {
public:
    struct Stats {
        double timeToLoadMs = 0.0;
        double timeToDecompressMs = 0.0;
        double timeToTransferMs = 0.0;
        double timeToCreateMeshMs = 0.0;
        uint totalTriangles = 0;
        QuadSet::Sizes sizes{};
    } stats;

    uint maxViews;
    float viewBoxSize;
    std::vector<ReferenceFrame> frames;

    QuadStreamReceiver(QuadSet& quadSet, uint maxViews);
    QuadStreamReceiver(QuadSet& quadSet, uint maxViews, float remoteFOV, float remoteFOVWide, float viewBoxSize);
    ~QuadStreamReceiver() = default;

    QuadMesh& getMesh(int view) { return meshes[view]; }
    PerspectiveCamera& getRemoteCamera(int view = 0) { return remoteCameras[view]; }
    void copyPoseToCamera(PerspectiveCamera& camera) {
        PerspectiveCamera& remoteCameraCenter = remoteCameras[0];
        camera.setViewMatrix(remoteCameraCenter.getViewMatrix());
        camera.setProjectionMatrix(remoteCameraCenter.getProjectionMatrix());
    }

    void updateViewBox(float viewBoxSize);

    void loadFromFiles(const Path& dataPath);

private:
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

    std::vector<PerspectiveCamera> remoteCameras;

    QuadMaterial quadMaterial;
    std::vector<Texture> colorTextures;
    std::vector<QuadMesh> meshes;

    std::unique_ptr<BS::thread_pool<>> threadPool;

    // Temporary buffers for decompression
    std::vector<char> uncompressedQuads, uncompressedOffsets;
};

} // namespace quasar

#endif // QUADSTREAM_RECEIVER_H
