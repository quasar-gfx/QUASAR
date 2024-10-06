#ifndef BC4_DEPTH_VIDEO_TEXTURE_H
#define BC4_DEPTH_VIDEO_TEXTURE_H

#include <iostream>
#include <iomanip>
#include <deque>
#include <Texture.h>
#include <Networking/DataReceiverTCP.h>
#include <Utils/TimeUtils.h>
#include <CameraPose.h>
#include <Buffer.h>

struct Block {
    float max;
    float min;
    uint32_t arr[6];
};

class BC4DepthVideoTexture : public Texture, public DataReceiverTCP {
public:
    std::string streamerURL;
    struct Stats {
        float timeToReceiveMs = -1.0f;
        float bitrateMbps = -1.0f;
    } stats;

    BC4DepthVideoTexture(const TextureDataCreateParams &params, std::string streamerURL);

    void setMaxQueueSize(unsigned int maxQueueSize) {
        this->maxQueueSize = maxQueueSize;
    }

    float getFrameRate() {
        return 1.0f / timeutils::millisToSeconds(stats.timeToReceiveMs);
    }

    pose_id_t draw(pose_id_t poseID = -1);
    pose_id_t getLatestPoseID();

    const Buffer<Block>& getCompressedBuffer() const { return bc4CompressedBuffer; }
    glm::uvec2 getDepthMapSize() const { return glm::uvec2(width, height); }

private:
    pose_id_t prevPoseID = -1;
    unsigned int maxQueueSize = 10;
    std::mutex m;
    struct FrameData {
        pose_id_t poseID;
        std::vector<uint8_t> buffer;
    };
    std::deque<FrameData> depthFrames;
    Buffer<Block> bc4CompressedBuffer;
    size_t compressedSize;

    void onDataReceived(const std::vector<uint8_t>& data) override;

    void debugPrintData(const std::vector<uint8_t>& data, size_t bytesToPrint = 64) {
        std::cout << "Received data (first " << bytesToPrint << " bytes):" << std::endl;
        for (size_t i = 0; i < std::min(data.size(), bytesToPrint); ++i) {
            std::cout << std::setw(2) << std::setfill('0') << std::hex << static_cast<int>(data[i]) << " ";
            if ((i + 1) % 16 == 0) std::cout << std::endl;
        }
        std::cout << std::dec << std::endl;
    }
};

#endif // BC4_DEPTH_VIDEO_TEXTURE_H
