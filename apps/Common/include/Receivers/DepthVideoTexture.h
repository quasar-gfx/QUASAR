#ifndef DEPTH_VIDEO_TEXTURE_H
#define DEPTH_VIDEO_TEXTURE_H

#include <deque>
#include <mutex>

#include <Texture.h>
#include <Networking/DataReceiverTCP.h>
#include <Utils/TimeUtils.h>
#include <CameraPose.h>

namespace quasar {

class DepthVideoTexture : public Texture, public DataReceiverTCP {
public:
    struct Stats {
        double receiveTimeMs = 0.0;
        double bitrateMbps = 0.0;
    } stats;

    DepthVideoTexture(const TextureDataCreateParams& params, std::string streamerURL);

    void setMaxQueueSize(size_t maxQueueSize) {
        this->maxQueueSize = maxQueueSize;
    }

    float getFrameRate() {
        return 1.0f / timeutils::millisToSeconds(stats.receiveTimeMs);
    }

    pose_id_t draw(pose_id_t poseID = -1);
    pose_id_t getLatestPoseID();

private:
    pose_id_t prevPoseID = -1;
    size_t maxQueueSize = 10;

    std::mutex m;

    struct FrameData {
        pose_id_t poseID;
        std::vector<char> buffer;
    };

    std::deque<FrameData> frames;

    void onDataReceived(const std::vector<char>& data) override;
};

} // namespace quasar

#endif // DEPTH_VIDEO_TEXTURE_H
