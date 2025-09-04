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
    struct Params {
        uint32_t numLayers;
        float viewSphereDiameter;
        float wideFOV;
    };

    struct Header {
        pose_id_t poseID;
        QuadFrame::FrameType frameType;
        uint32_t cameraSize;
        Params params;
        uint32_t geometrySize;
    };

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

    VideoTexture atlasVideoTexture;

    QUASARReceiver(QuadSet& quadSet, uint maxLayers, const std::string& videoURL = "", const std::string& proxiesURL = "");
    QUASARReceiver(QuadSet& quadSet, uint maxLayers, float remoteFOV, float remoteFOVWide, const std::string& videoURL = "", const std::string& proxiesURL = "");
    ~QUASARReceiver() = default;

    QuadMesh& getMesh(int layer) { return meshes[layer]; }
    QuadMesh& getResidualMesh() { return residualFrameMesh; }
    PerspectiveCamera& getRemoteCamera() { return remoteCamera; }
    PerspectiveCamera& getRemoteCameraPrev() { return remoteCameraPrev; }
    PerspectiveCamera& getremoteCameraWideFOV() { return remoteCameraWideFOV; }
    void copyPoseToCamera(PerspectiveCamera& camera) {
        camera.setViewMatrix(remoteCamera.getViewMatrix());
        camera.setProjectionMatrix(remoteCamera.getProjectionMatrix());
    }

    void setViewSphereDiameter(float viewSphereDiameter) { this->viewSphereDiameter = viewSphereDiameter; }

    QuadFrame::FrameType loadFromFiles(const Path& dataPath);
    QuadFrame::FrameType loadFromMemory(const std::vector<char>& inputData);

    QuadFrame::FrameType recvData();

private:
    QuadSet& quadSet;
    PerspectiveCamera remoteCamera;
    PerspectiveCamera remoteCameraWideFOV;
    PerspectiveCamera remoteCameraPrev;

    std::vector<QuadMesh> meshes;
    QuadMesh residualFrameMesh;

    struct Frame {
        pose_id_t poseID;
        QuadFrame::FrameType frameType;
        Pose cameraPose;

        std::vector<std::vector<char>> uncompressedQuads, uncompressedOffsets;
        std::vector<char> uncompressedQuadsRevealed, uncompressedOffsetsRevealed;

        Frame(const glm::vec2& gBufferSize, int maxLayers, size_t maxProxiesPerMesh = MAX_PROXIES_PER_MESH)
            : frameType(QuadFrame::FrameType::NONE)
        {
            const size_t quadsBytes   = sizeof(uint) + maxProxiesPerMesh * sizeof(QuadMapDataPacked);
            const size_t offsetsBytes = static_cast<size_t>(gBufferSize.x * gBufferSize.y) * 4 * sizeof(uint16_t);
            for (int layer = 0; layer < maxLayers; layer++) {
                size_t adjustedQuadsBytes =
                    (layer == 0 || layer == maxLayers - 1) ? quadsBytes :
                        (layer == 1) ? quadsBytes / 4 : quadsBytes / 8;

                uncompressedQuads.emplace_back(std::vector<char>(adjustedQuadsBytes));
                uncompressedOffsets.emplace_back(std::vector<char>(offsetsBytes));
            }

            uncompressedQuadsRevealed.resize(quadsBytes);
            uncompressedOffsetsRevealed.resize(offsetsBytes);
        }
        ~Frame() = default;

        size_t decompressReferenceFrames(std::unique_ptr<BS::thread_pool<>>& threadPool,
                                         std::vector<ReferenceFrame>& referenceFrames) {
            // Decompress hidden layer and wide fov proxies (asynchronous)
            std::vector<std::future<size_t>> futures;
            futures.reserve(referenceFrames.size() * 2);
            for (int layer = 0; layer < referenceFrames.size(); layer++) {
                futures.emplace_back(threadPool->submit_task([&, layer]() {
                    return referenceFrames[layer].decompressDepthOffsets(uncompressedOffsets[layer]);
                }));
                futures.emplace_back(threadPool->submit_task([&, layer]() {
                    return referenceFrames[layer].decompressQuads(uncompressedQuads[layer]);
                }));
            }

            size_t outputSize = 0;
            for (auto& f : futures) outputSize += f.get();
            return outputSize;
        }

        size_t decompressReferenceAndResidualFrames(std::unique_ptr<BS::thread_pool<>>& threadPool,
                                                    std::vector<ReferenceFrame>& referenceFrames,
                                                    ResidualFrame& residualFrame) {
            // Decompress hidden layer and wide fov proxies (asynchronous)
            std::vector<std::future<size_t>> futures;
            futures.reserve((referenceFrames.size() - 1) * 2 + 4);
            futures.emplace_back(threadPool->submit_task([&]() {
                return residualFrame.decompressUpdatedDepthOffsets(uncompressedOffsets[0]);
            }));
            futures.emplace_back(threadPool->submit_task([&]() {
                return residualFrame.decompressRevealedDepthOffsets(uncompressedOffsetsRevealed);
            }));
            futures.emplace_back(threadPool->submit_task([&]() {
                return residualFrame.decompressUpdatedQuads(uncompressedQuads[0]);
            }));
            futures.emplace_back(threadPool->submit_task([&]() {
                return residualFrame.decompressRevealedQuads(uncompressedQuadsRevealed);
            }));
            for (int layer = 1; layer < referenceFrames.size(); layer++) {
                futures.emplace_back(threadPool->submit_task([&, layer]() {
                    return referenceFrames[layer].decompressDepthOffsets(uncompressedOffsets[layer]);
                }));
                futures.emplace_back(threadPool->submit_task([&, layer]() {
                    return referenceFrames[layer].decompressQuads(uncompressedQuads[layer]);
                }));
            }

            size_t outputSize = 0;
            for (auto& f : futures) outputSize += f.get();
            return outputSize;
        }
    };

    std::mutex m;
    std::condition_variable cv;
    std::shared_ptr<Frame> frameInUse;
    std::shared_ptr<Frame> framePending;
    std::shared_ptr<Frame> frameFree;

    std::vector<ReferenceFrame> referenceFrames;
    ResidualFrame residualFrame;

    std::unique_ptr<BS::thread_pool<>> threadPool;

    std::vector<char> geometryData;

    inline const PerspectiveCamera& getCameraToUse(int layer) const {
        return (layer == maxLayers - 1) ? remoteCameraWideFOV : remoteCamera;
    }

    void onDataReceived(const std::vector<char>& data) override;
    QuadFrame::FrameType loadFromFrame(std::shared_ptr<Frame> frame);
};

} // namespace quasar

#endif // QUASAR_RECEIVER_H
