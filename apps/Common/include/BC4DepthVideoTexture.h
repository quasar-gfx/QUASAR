#ifndef BC4_DEPTH_VIDEO_TEXTURE_H
#define BC4_DEPTH_VIDEO_TEXTURE_H

#include <iomanip>
#include <deque>

#include <Buffer.h>
#include <Texture.h>
#include <Networking/DataReceiverTCP.h>
#include <Utils/TimeUtils.h>

#include <CameraPose.h>

#include <Codec/BC4.h>
#include <Codec/ZSTDCodec.h>

namespace quasar {

class BC4DepthVideoTexture : public Texture, public DataReceiverTCP {
public:
    Buffer bc4CompressedBuffer;

    struct ReceiverStats {
        double timeToReceiveMs = 0.0;
        double timeToDecompressMs = 0.0;
        double bitrateMbps = 0.0;
        double compressionRatio = 0.0;
    };

    ReceiverStats stats;

    BC4DepthVideoTexture(const TextureDataCreateParams &params, std::string streamerURL);

    void setMaxQueueSize(uint maxQueueSize) {
        this->maxQueueSize = maxQueueSize;
    }

    float getFrameRate() {
        return 1.0f / timeutils::millisToSeconds(stats.timeToReceiveMs);
    }

    pose_id_t draw(pose_id_t poseID = -1);
    pose_id_t getLatestPoseID();

private:
    time_t prevTime = timeutils::getTimeMicros();
    pose_id_t prevPoseID = -1;

    uint maxQueueSize = 10;
    std::mutex m;

    struct FrameData {
        pose_id_t poseID;
        std::vector<char> buffer;
    };
    std::deque<FrameData> depthFrames;
    size_t compressedSize;

    ZSTDCodec codec;

    void onDataReceived(const std::vector<char>& data) override;
};

} // namespace quasar

#endif // BC4_DEPTH_VIDEO_TEXTURE_H
