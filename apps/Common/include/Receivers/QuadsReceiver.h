#ifndef QUADS_RECEIVER_H
#define QUADS_RECEIVER_H

#include <BS_thread_pool/BS_thread_pool.hpp>

#include <Path.h>
#include <CameraPose.h>
#include <Quads/QuadFrames.h>
#include <Quads/QuadMesh.h>
#include <Networking/DataReceiverTCP.h>
#include <Receivers/VideoTexture.h>
#include <Codecs/AlphaCodec.h>

namespace quasar {

class QuadsReceiver : public DataReceiverTCP {
public:
    struct Header {
        pose_id_t poseID;
        QuadFrame::FrameType frameType;
        uint32_t cameraSize;
        uint32_t alphaSize;
        uint32_t geometrySize;

        size_t getSize() const { return sizeof(Header) + cameraSize + alphaSize + geometrySize; }
    };

    struct Stats {
        double loadTimeMs = 0.0;
        double decompressTimeMs = 0.0;
        double transferTimeMs = 0.0;
        double createMeshTimeMs = 0.0;
        size_t totalTriangles = 0;
        QuadSet::Sizes sizes{};
    } stats;

    std::string proxiesURL;
    std::string videoURL;

    VideoTexture videoAtlasTexture;
    Texture alphaAtlasTexture;

    QuadsReceiver(QuadSet& quadSet, const std::string& videoURL = "", const std::string& proxiesURL = "");
    QuadsReceiver(QuadSet& quadSet, float remoteFOV, const std::string& videoURL = "", const std::string& proxiesURL = "");
    ~QuadsReceiver() = default;

    QuadMesh& getReferenceMesh() { return referenceFrameMesh; }
    QuadMesh& getResidualMesh() { return residualFrameMesh; }
    PerspectiveCamera& getRemoteCamera() { return remoteCamera; }
    PerspectiveCamera& getRemoteCameraPrev() { return remoteCameraPrev; }
    void copyPoseToCamera(PerspectiveCamera& camera) { frameInUse->cameraPose.copyPoseToCamera(camera); }

    QuadFrame::FrameType loadFromFiles(const Path& dataPath);
    QuadFrame::FrameType loadFromMemory(const std::vector<char>& inputData);

    QuadFrame::FrameType recvData();

private:
    QuadSet& quadSet;
    PerspectiveCamera remoteCamera;
    PerspectiveCamera remoteCameraPrev;

    QuadMesh referenceFrameMesh;
    QuadMesh residualFrameMesh;

    struct Frame {
        pose_id_t poseID;
        QuadFrame::FrameType frameType;

        Pose cameraPose;

        std::vector<char> alphaData;
        std::vector<char> uncompressedQuads, uncompressedOffsets;
        std::vector<char> uncompressedQuadsRevealed, uncompressedOffsetsRevealed;

        Frame(const glm::vec2& gBufferSize)
            : frameType(QuadFrame::FrameType::NONE)
        {
            const size_t quadsBytes = sizeof(uint32_t) + MAX_PROXIES_PER_MESH * sizeof(QuadMapDataPacked);
            const size_t offsetsBytes = static_cast<size_t>(gBufferSize.x * gBufferSize.y) * 4 * sizeof(uint16_t);

            uncompressedQuads.resize(quadsBytes);
            uncompressedOffsets.resize(offsetsBytes);

            uncompressedQuadsRevealed.resize(quadsBytes / 4); // Residual frame usually has less quads
            uncompressedOffsetsRevealed.resize(offsetsBytes);

            alphaData.resize(2 * gBufferSize.x * gBufferSize.y);
        }
        ~Frame() = default;

        size_t decompressReferenceFrame(std::unique_ptr<BS::thread_pool<>>& threadPool, ReferenceFrame& referenceFrame) {
            // Decompress proxies (asynchronous)
            std::vector<std::future<size_t>> futures;
            futures.emplace_back(threadPool->submit_task([&]() {
                return referenceFrame.decompressDepthOffsets(uncompressedOffsets);
            }));
            futures.emplace_back(threadPool->submit_task([&]() {
                return referenceFrame.decompressQuads(uncompressedQuads);
            }));

            size_t outputSize = 0;
            for (auto& f : futures) outputSize += f.get();
            return outputSize;
        }

        size_t decompressResidualFrame(std::unique_ptr<BS::thread_pool<>>& threadPool, ResidualFrame& residualFrame) {
            // Decompress proxies (asynchronous)
            std::vector<std::future<size_t>> futures;
            futures.emplace_back(threadPool->submit_task([&]() {
                return residualFrame.decompressUpdatedDepthOffsets(uncompressedOffsets);
            }));
            futures.emplace_back(threadPool->submit_task([&]() {
                return residualFrame.decompressRevealedDepthOffsets(uncompressedOffsetsRevealed);
            }));
            futures.emplace_back(threadPool->submit_task([&]() {
                return residualFrame.decompressUpdatedQuads(uncompressedQuads);
            }));
            futures.emplace_back(threadPool->submit_task([&]() {
                return residualFrame.decompressRevealedQuads(uncompressedQuadsRevealed);
            }));

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

    AlphaCodec alphaCodec;

    bool waitUntilReferenceFrame = false;

    ReferenceFrame referenceFrame;
    ResidualFrame residualFrame;

    std::unique_ptr<BS::thread_pool<>> threadPool;

    void onDataReceived(const std::vector<char>& data) override;
    QuadFrame::FrameType reconstructFrame(std::shared_ptr<Frame> frame);
};

} // namespace quasar

#endif // QUADS_RECEIVER_H
