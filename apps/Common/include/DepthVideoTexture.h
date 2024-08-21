#ifndef DEPTH_RECEIVER_H
#define DEPTH_RECEIVER_H

#include <deque>

#include <Texture.h>

#include <Networking/DataReceiverTCP.h>
#include <Utils/TimeUtils.h>
#include <CameraPose.h>

class DepthVideoTexture : public Texture {
public:
    std::string streamerURL;

    struct Stats {
        float timeToReceiveMs = -1.0f;
        float bitrateMbps = -1.0f;
    } stats;

    DepthVideoTexture(const TextureDataCreateParams &params, std::string streamerURL)
            : streamerURL(streamerURL)
            , receiver(streamerURL)
            , Texture(params) { }

    void setMaxQueueSize(unsigned int maxQueueSize) {
        this->maxQueueSize = maxQueueSize;
    }

    float getFrameRate() {
        return 1.0f / timeutils::millisToSeconds(stats.timeToReceiveMs);
    }

    pose_id_t draw(pose_id_t poseID = -1);

private:
    pose_id_t prevPoseID = -1;
    unsigned int maxQueueSize = 10;

    struct FrameData {
        pose_id_t poseID;
        std::vector<uint8_t> buffer;
    };

    DataReceiverTCP receiver;

    std::deque<FrameData> datas;
};

#endif // DEPTH_RECEIVER_H
