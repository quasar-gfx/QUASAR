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
#include <Codecs/AlphaCodec.h>

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
        Params params;
        uint32_t cameraSize;
        uint32_t alphaSize;
        uint32_t geometrySize;

        size_t getSize() const { return sizeof(Header) + cameraSize + alphaSize + geometrySize; }
    };

    struct Stats {
        size_t totalTriangles = 0;
        double loadTimeMs = 0.0;
        double decompressTimeMs = 0.0;
        double transferTimeMs = 0.0;
        double createMeshTimeMs = 0.0;
        QuadSet::Sizes sizes{};
    } stats;

    std::string proxiesURL;
    std::string videoURL;

    uint maxLayers;
    float viewSphereDiameter;

    VideoTexture videoAtlasTexture;
    Texture alphaAtlasTexture;

    QUASARReceiver(QuadSet& quadSet, uint maxLayers, const std::string& videoURL = "", const std::string& proxiesURL = "");
    QUASARReceiver(QuadSet& quadSet, uint maxLayers, float remoteFOV, float remoteFOVWide, const std::string& videoURL = "", const std::string& proxiesURL = "");
    ~QUASARReceiver() = default;

    QuadMesh& getMesh(int layer) { return meshes[layer]; }
    QuadMesh& getResidualMesh() { return residualFrameMesh; }
    PerspectiveCamera& getRemoteCamera() { return remoteCamera; }
    PerspectiveCamera& getRemoteCameraPrev() { return remoteCameraPrev; }
    PerspectiveCamera& getremoteCameraWideFOV() { return remoteCameraWideFOV; }
    void copyPoseToCamera(PerspectiveCamera& camera);

    void setDrawState(QuadMesh::DrawState drawState);
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

    struct BufferPool {
        std::vector<char> alphaData;
        std::vector<std::vector<char>> uncompressedQuads, uncompressedOffsets;
        std::vector<char> uncompressedQuadsRevealed, uncompressedOffsetsRevealed;

        BufferPool(const glm::vec2& gBufferSize, int maxLayers) {
            uncompressedQuads.resize(maxLayers);
            uncompressedOffsets.resize(maxLayers);

            const size_t quadsBytes   = sizeof(uint32_t) + MAX_PROXIES_PER_MESH * sizeof(QuadMapDataPacked);
            const size_t offsetsBytes = static_cast<size_t>(gBufferSize.x * gBufferSize.y) * 4 * sizeof(uint16_t);

            for (int layer = 0; layer < maxLayers; layer++) {
                uncompressedQuads[layer].resize(quadsBytes);
                uncompressedOffsets[layer].resize(offsetsBytes);
            }

            uncompressedQuadsRevealed.resize(quadsBytes / 4); // Residual frame usually has less quads
            uncompressedOffsetsRevealed.resize(offsetsBytes);

            alphaData.resize(2 * gBufferSize.x * 3 * gBufferSize.y);
        }
    };

    struct Frame {
        pose_id_t poseID = -1;
        QuadFrame::FrameType frameType;
        Pose cameraPose;
        BufferPool& bufferPool;

        Frame(BufferPool& bufferPool)
            : frameType(QuadFrame::FrameType::NONE)
            , bufferPool(bufferPool)
        {}
        ~Frame() = default;

        size_t decompressReferenceHiddenLayersWideFOV(std::unique_ptr<BS::thread_pool<>>& threadPool,
                                                      std::vector<ReferenceFrame>& referenceFrames) {
            std::vector<std::future<size_t>> futures;
            for (int layer = 0; layer < referenceFrames.size(); layer++) {
                futures.emplace_back(threadPool->submit_task([&, layer]() {
                    return referenceFrames[layer].decompressDepthOffsets(bufferPool.uncompressedOffsets[layer]);
                }));
                futures.emplace_back(threadPool->submit_task([&, layer]() {
                    return referenceFrames[layer].decompressQuads(bufferPool.uncompressedQuads[layer]);
                }));
            }

            size_t outputSize = 0;
            for (auto& f : futures) outputSize += f.get();
            return outputSize;
        }

        size_t decompressResidualFrame(std::unique_ptr<BS::thread_pool<>>& threadPool,
                                       ResidualFrame& residualFrame) {
            std::vector<std::future<size_t>> futures;
            futures.emplace_back(threadPool->submit_task([&]() {
                return residualFrame.decompressUpdatedDepthOffsets(bufferPool.uncompressedOffsets[0]);
            }));
            futures.emplace_back(threadPool->submit_task([&]() {
                return residualFrame.decompressRevealedDepthOffsets(bufferPool.uncompressedOffsetsRevealed);
            }));
            futures.emplace_back(threadPool->submit_task([&]() {
                return residualFrame.decompressUpdatedQuads(bufferPool.uncompressedQuads[0]);
            }));
            futures.emplace_back(threadPool->submit_task([&]() {
                return residualFrame.decompressRevealedQuads(bufferPool.uncompressedQuadsRevealed);
            }));

            size_t outputSize = 0;
            for (auto& f : futures) outputSize += f.get();
            return outputSize;
        }

        size_t decompressHiddenLayersWideFOV(std::unique_ptr<BS::thread_pool<>>& threadPool,
                                                    std::vector<ReferenceFrame>& referenceFrames) {
            std::vector<std::future<size_t>> futures;
            for (int layer = 1; layer < referenceFrames.size(); layer++) {
                futures.emplace_back(threadPool->submit_task([&, layer]() {
                    return referenceFrames[layer].decompressDepthOffsets(bufferPool.uncompressedOffsets[layer]);
                }));
                futures.emplace_back(threadPool->submit_task([&, layer]() {
                    return referenceFrames[layer].decompressQuads(bufferPool.uncompressedQuads[layer]);
                }));
            }

            size_t outputSize = 0;
            for (auto& f : futures) outputSize += f.get();
            return outputSize;
        }

        size_t decompressResidualHiddenLayersWideFOV(std::unique_ptr<BS::thread_pool<>>& threadPool,
                                                    std::vector<ReferenceFrame>& referenceFrames,
                                                    ResidualFrame& residualFrame) {
            std::vector<std::future<size_t>> futures;
            futures.reserve(4 + 2 * (referenceFrames.size() - 1));

            futures.emplace_back(threadPool->submit_task([&]() {
                return residualFrame.decompressUpdatedDepthOffsets(bufferPool.uncompressedOffsets[0]);
            }));
            futures.emplace_back(threadPool->submit_task([&]() {
                return residualFrame.decompressRevealedDepthOffsets(bufferPool.uncompressedOffsetsRevealed);
            }));
            futures.emplace_back(threadPool->submit_task([&]() {
                return residualFrame.decompressUpdatedQuads(bufferPool.uncompressedQuads[0]);
            }));
            futures.emplace_back(threadPool->submit_task([&]() {
                return residualFrame.decompressRevealedQuads(bufferPool.uncompressedQuadsRevealed);
            }));

            for (int layer = 1; layer < referenceFrames.size(); layer++) {
                futures.emplace_back(threadPool->submit_task([&, layer]() {
                    return referenceFrames[layer].decompressDepthOffsets(bufferPool.uncompressedOffsets[layer]);
                }));
                futures.emplace_back(threadPool->submit_task([&, layer]() {
                    return referenceFrames[layer].decompressQuads(bufferPool.uncompressedQuads[layer]);
                }));
            }

            size_t outputSize = 0;
            for (auto& f : futures) outputSize += f.get();
            return outputSize;
        }
    };

    BufferPool bufferPool;

    std::mutex m;
    std::condition_variable cv;
    std::shared_ptr<Frame> frameInUse;
    std::shared_ptr<Frame> framePending;
    std::shared_ptr<Frame> frameFree;

    AlphaCodec alphaCodec;

    bool waitUntilReferenceFrame = false;

    std::vector<ReferenceFrame> referenceFrames;
    ResidualFrame residualFrame;

    std::unique_ptr<BS::thread_pool<>> threadPool;

    inline const PerspectiveCamera& getCameraToUse(int layer) const {
        return (layer == maxLayers - 1) ? remoteCameraWideFOV : remoteCamera;
    }

    void onDataReceived(const std::vector<char>& data) override;
    QuadFrame::FrameType reconstructFrame(std::shared_ptr<Frame> frame);
};

} // namespace quasar

#endif // QUASAR_RECEIVER_H
