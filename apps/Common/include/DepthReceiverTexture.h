#ifndef IMAGE_RECEIVER_H
#define IMAGE_RECEIVER_H

#include <deque>

#include <DataReceiver.h>

#include <Texture.h>

#include <CameraPose.h>

class DepthReceiverTexture : public Texture {
public:
    std::string streamerURL;

    unsigned int maxQueueSize = 10;

    explicit DepthReceiverTexture(const TextureCreateParams &params, std::string streamerURL)
            : streamerURL(streamerURL)
            , receiver(streamerURL)
            , Texture(params) { }

    void setMaxQueueSize(unsigned int maxQueueSize) {
        this->maxQueueSize = maxQueueSize;
    }

    pose_id_t draw(pose_id_t poseID = -1) {
        std::vector<uint8_t> data = receiver.recv();
        if (data.empty()) {
            return -1;
        }

        datas.push_back(std::move(data));

        if (datas.size() > maxQueueSize) {
            datas.pop_front();
        }

        std::vector<uint8_t> res;
        bool found = false;

        if (poseID == -1) {
            res = std::move(datas.front());
            datas.pop_front();
            found = true;
        }
        else {
            for (auto it = datas.begin(); it != datas.end(); ++it) {
                if (*reinterpret_cast<pose_id_t*>(it->data()) == poseID) {
                    res = std::move(*it);
                    datas.erase(it);
                    found = true;
                    break;
                }
            }
        }

        if (!found) {
            return -1;
        }

        pose_id_t resPoseID;
        memcpy(&resPoseID, res.data(), sizeof(pose_id_t));

        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED, GL_UNSIGNED_SHORT, res.data() + sizeof(pose_id_t));

        return resPoseID;
    }

private:
    DataReceiverTCP receiver;

    std::deque<std::vector<uint8_t>> datas;
};

#endif // IMAGE_RECEIVER_H
