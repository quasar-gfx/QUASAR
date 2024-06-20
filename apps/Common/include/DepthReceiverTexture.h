#ifndef IMAGE_RECEIVER_H
#define IMAGE_RECEIVER_H

#include <deque>

#include <DataReceiver.h>

#include <Texture.h>

#include <CameraPose.h>

class DepthReceiverTexture : public Texture {
public:
    std::string streamerURL;

    DataReceiverTCP* receiver;

    int imageSize;

    unsigned int maxQueueSize = 10;

    explicit DepthReceiverTexture(const TextureCreateParams &params, std::string streamerURL)
            : streamerURL(streamerURL)
            , imageSize(sizeof(pose_id_t) + params.width * params.height * sizeof(GLushort))
            , Texture(params) {
        receiver = new DataReceiverTCP(streamerURL, imageSize);
    }
    ~DepthReceiverTexture() {
        delete receiver;
    }

    void setMaxQueueSize(unsigned int maxQueueSize) {
        this->maxQueueSize = maxQueueSize;
    }

    pose_id_t draw(pose_id_t poseID = -1) {
        uint8_t* data = receiver->recv();
        if (data == nullptr) {
            return -1;
        }

        datas.push_back(data);

        if (datas.size() > maxQueueSize) {
            uint8_t* dataToFree = datas.front();
            delete[] dataToFree;
            datas.pop_front();
        }

        uint8_t* res = nullptr;
        if (poseID == -1) {
            res = datas.front();
            datas.pop_front();
        }
        else {
            for (auto it = datas.begin(); it != datas.end(); it++) {
                if (reinterpret_cast<uintptr_t>(*it) == poseID) {
                    res = *it;
                    break;
                }
            }
        }

        if (res == nullptr) {
            return -1;
        }

        pose_id_t resPoseID;
        memcpy(&resPoseID, res, sizeof(pose_id_t));

        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED, GL_UNSIGNED_SHORT, res + sizeof(pose_id_t));

        return resPoseID;
    }

private:
    std::deque<uint8_t*> datas;
};

#endif // IMAGE_RECEIVER_H
