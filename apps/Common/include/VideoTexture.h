#ifndef VIDEO_TEXTURE_H
#define VIDEO_TEXTURE_H

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/time.h>
#include <libavutil/imgutils.h>
}

#include <iostream>
#include <deque>
#include <atomic>
#include <thread>
#include <mutex>

#include <Utils/TimeUtils.h>

#include <Texture.h>

#include <CameraPose.h>

class VideoTexture : public Texture {
public:
    std::string videoURL = "127.0.0.1:12345";

    struct Stats {
        float timeToReceiveMs = -1.0f;
        float timeToDecodeMs = -1.0f;
        float timeToResizeMs = -1.0f;
        float totalTimeToReceiveMs = -1.0f;
        float bitrateMbps = -1.0f;
    } stats;

    unsigned int videoWidth, videoHeight;

    explicit VideoTexture(const TextureCreateParams &params, const std::string &videoURL);
    ~VideoTexture() {
        cleanup();
    }

    void cleanup();

    pose_id_t draw(pose_id_t poseID = -1);
    pose_id_t getLatestPoseID();

    void setMaxQueueSize(unsigned int maxQueueSize) {
        this->maxQueueSize = maxQueueSize;
    }

    float getFrameRate() {
        return 1.0f / timeutils::millisToSeconds(stats.totalTimeToReceiveMs);
    }

private:
    pose_id_t prevPoseID = -1;
    uint64_t framesReceived = 0;
    unsigned int maxQueueSize = 10;

    AVPixelFormat openglPixelFormat = AV_PIX_FMT_RGB24;

    AVFormatContext* inputFormatCtx = avformat_alloc_context();
    AVCodecContext* codecCtx = nullptr;

    int videoStreamIndex = -1;

    struct SwsContext* swsCtx = nullptr;

    AVFrame* frame = av_frame_alloc();
    AVPacket* packet = av_packet_alloc();

    std::atomic_bool videoReady = false;
    bool shouldTerminate = false;

    std::thread videoReceiverThread;
    std::mutex m;

    struct FrameData {
        pose_id_t poseID;
        AVFrame* frame;
        uint8_t* buffer;

        void free() {
            av_frame_free(&frame);
            delete[] buffer;
        }
    };

    std::deque<FrameData> frames;

    void receiveVideo();

    int initFFMpeg();
};

#endif // VIDEO_TEXTURE_H
