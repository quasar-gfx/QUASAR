#ifndef BC4_DEPTH_VIDEO_TEXTURE_H
#define BC4_DEPTH_VIDEO_TEXTURE_H

#include <iomanip>
#include <deque>

#include <Buffer.h>
#include <Texture.h>
#include <Networking/DataReceiverTCP.h>
#include <Utils/TimeUtils.h>

#include <CameraPose.h>

#include <Codec/ZSTDCodec.h>

namespace quasar {

class BC4DepthVideoTexture : public Texture, public DataReceiverTCP {
public:
    struct Block {
        float max;
        float min;
        uint32_t data[6];
    };
    Buffer bc4CompressedBuffer;

    std::string streamerURL;

    struct ReceiverStats {
        double timeToReceiveMs = 0.0f;
        double timeToDecompressMs = 0.0f;
        double bitrateMbps = 0.0f;
        double compressionRatio = 0.0f;
    };

    ReceiverStats stats;

    BC4DepthVideoTexture(const TextureDataCreateParams &params, std::string streamerURL);

    void setMaxQueueSize(unsigned int maxQueueSize) {
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

    unsigned int maxQueueSize = 10;
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
