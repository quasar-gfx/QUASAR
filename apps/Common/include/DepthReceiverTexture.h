#ifndef DEPTH_RECEIVER_H
#define DEPTH_RECEIVER_H

#include <deque>

#include <Texture.h>

#include <Networking/DataReceiverTCP.h>

#include <CameraPose.h>

class DepthReceiverTexture : public Texture {
public:
    std::string streamerURL;

    struct Stats {
        float timeToReceiveMs = -1.0f;
        float bitrateMbps = -1.0f;
    } stats;

    explicit DepthReceiverTexture(const TextureCreateParams &params, std::string streamerURL)
            : streamerURL(streamerURL)
            , receiver(streamerURL)
            , Texture(params) { }

    void setMaxQueueSize(unsigned int maxQueueSize) {
        this->maxQueueSize = maxQueueSize;
    }

    pose_id_t draw(pose_id_t poseID = -1);

private:
    unsigned int maxQueueSize = 10;

    DataReceiverTCP receiver;

    std::deque<std::vector<uint8_t>> datas;
};

#endif // DEPTH_RECEIVER_H
