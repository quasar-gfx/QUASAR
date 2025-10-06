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
    uint width, height;

    Buffer bc4CompressedBuffer;

    struct ReceiverStats {
        double receiveTimeMs = 0.0;
        double decompressTimeMs = 0.0;
        double bitrateMbps = 0.0;
        double compressionRatio = 0.0;
    };

    ReceiverStats stats;

    BC4DepthVideoTexture(const TextureDataCreateParams& params, std::string streamerURL = "");
    ~BC4DepthVideoTexture() = default;

    pose_id_t getLatestPoseID();
    float getFrameRate() { return 1.0f / timeutils::millisToSeconds(stats.receiveTimeMs); }

    void setMaxQueueSize(size_t maxQueueSize) { this->maxQueueSize = maxQueueSize; }

    pose_id_t draw(pose_id_t poseID = -1);

    void loadFromFile(const Path& dataPath);

private:
    time_t prevTime = timeutils::getTimeMicros();
    pose_id_t prevPoseID = -1;
    size_t maxQueueSize = 6;

    std::mutex m;

    struct FrameData {
        pose_id_t poseID;
        std::vector<char> buffer;
    };
    std::deque<FrameData> frames;
    size_t compressedSize;

    std::vector<char> decompressedData;

    ZSTDCodec codec;

    void onDataReceived(const std::vector<char>& data) override;
};

} // namespace quasar

#endif // BC4_DEPTH_VIDEO_TEXTURE_H
