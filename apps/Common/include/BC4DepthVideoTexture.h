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

#include <lz4_stream/lz4_stream.h>

class BC4DepthVideoTexture : public Texture, public DataReceiverTCP {
public:
    struct Block {
        float max;
        float min;
        uint32_t data[6];
    };
    Buffer<Block> bc4CompressedBuffer;

    std::string streamerURL;

    // struct Stats {  
    //     float timeToReceiveMs = -1.0f;
    //     float bitrateMbps = -1.0f;
    // } stats;

    struct ReceiverStats {
        float timeToReceiveMs = -1.0f;
        float timeToDecompressMs = -1.0f;
        float bitrateMbps = -1.0f;
        float lz4CompressionRatio = -1.0f;
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
    pose_id_t prevPoseID = -1;
    unsigned int maxQueueSize = 10;
    std::mutex m;
    struct FrameData {
        pose_id_t poseID;
        std::vector<uint8_t> buffer;
    };
    std::deque<FrameData> depthFrames;
    size_t compressedSize;

    void onDataReceived(const std::vector<uint8_t>& data) override;
};

#endif // BC4_DEPTH_VIDEO_TEXTURE_H
