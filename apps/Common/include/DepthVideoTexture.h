#ifndef DEPTH_VIDEO_TEXTURE_H
#define DEPTH_VIDEO_TEXTURE_H

#include <deque>

#include <Texture.h>

#include <Networking/DataReceiverTCP.h>
#include <Utils/TimeUtils.h>
#include <CameraPose.h>

namespace quasar {

class DepthVideoTexture : public Texture, public DataReceiverTCP {
public:
    struct Stats {
        double timeToReceiveMs = 0.0;
        double bitrateMbps = 0.0;
    } stats;

    DepthVideoTexture(const TextureDataCreateParams &params, std::string streamerURL)
            : DataReceiverTCP(streamerURL, false)
            , Texture(params) { }

    void setMaxQueueSize(uint maxQueueSize) {
        this->maxQueueSize = maxQueueSize;
    }

    float getFrameRate() {
        return 1.0f / timeutils::millisToSeconds(stats.timeToReceiveMs);
    }

    pose_id_t draw(pose_id_t poseID = -1);
    pose_id_t getLatestPoseID();

private:
    pose_id_t prevPoseID = -1;
    uint maxQueueSize = 10;

    std::mutex m;

    struct FrameData {
        pose_id_t poseID;
        std::vector<char> buffer;
    };

    std::deque<FrameData> depthFrames;

    void onDataReceived(const std::vector<char>& data) override;
};

} // namespace quasar

#endif // DEPTH_VIDEO_TEXTURE_H
