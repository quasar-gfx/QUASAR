#ifndef BC4_DEPTH_VIDEO_TEXTURE_H
#define BC4_DEPTH_VIDEO_TEXTURE_H

#include <deque>
#include <Texture.h>
#include <Networking/DataReceiverTCP.h>
#include <Utils/TimeUtils.h>
#include <CameraPose.h>

struct Block {
    float max; // 32 - unit32
    float min;
    uint32_t arr[6];
};

class BC4DepthVideoTexture : public Texture, public DataReceiverTCP {
public:
    std::string streamerURL;
    struct Stats {
        float timeToReceiveMs = -1.0f;
        float timeToDecompressMs = -1.0f;
        float bitrateMbps = -1.0f;
    } stats;

    BC4DepthVideoTexture(const TextureDataCreateParams &params, std::string streamerURL);
    ~BC4DepthVideoTexture();

    void setMaxQueueSize(unsigned int maxQueueSize) {
        this->maxQueueSize = maxQueueSize;
    }

    float getFrameRate() {
        return 1.0f / timeutils::millisToSeconds(stats.timeToReceiveMs);
    }

    pose_id_t draw(pose_id_t poseID = -1);
    pose_id_t getLatestPoseID();

private:
    pose_id_t prevPoseID = -1;
    unsigned int maxQueueSize = 10;
    std::mutex m;
    struct FrameData {
        pose_id_t poseID;
        std::vector<uint8_t> buffer;
    };
    std::deque<FrameData> depthFrames;

    void onDataReceived(const std::vector<uint8_t>& data) override;

    // BC4 decompression related
    void decompressBC4(const std::vector<uint8_t>& compressedData);
    unsigned int compressedSize;
    GLuint bc4DecompressComputeShader;
    GLuint bc4CompressedBuffer;
    GLuint bc4DecompressedBuffer;
};

#endif // BC4_DEPTH_VIDEO_TEXTURE_H