#ifndef VIDEO_RECEIVER_H
#define VIDEO_RECEIVER_H

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/time.h>
#include <libavutil/imgutils.h>
}

#include <iostream>
#include <thread>
#include <mutex>

#include <Texture.h>

#include <CameraPose.h>

#define MICROSECONDS_IN_SECOND 1e6f
#define MICROSECONDS_IN_MILLISECOND 1e3f

#define MBPS_TO_BPS 1e6f

class VideoTexture : public Texture {
public:
    std::string videoURL = "localhost:12345";

    unsigned int width, height;

    int frameReceived = 0;

    struct Stats {
        float timeToReceiveFrame = -1.0f;
        float timeToDecode = -1.0f;
        float timeToResize = -1.0f;
        float totalTimeToReceiveFrame = -1.0f;
    } stats;

    explicit VideoTexture(const TextureCreateParams &params, const std::string &videoURL);
    ~VideoTexture() {
        cleanup();
    }

    void cleanup();

    pose_id_t draw();

    pose_id_t getPoseID() {
        if (frameRGB == nullptr) {
            return -1;
        }
        return static_cast<pose_id_t>(reinterpret_cast<uintptr_t>(frameRGB->opaque));
    }

    float getFrameRate() {
        return 1.0f / stats.totalTimeToReceiveFrame;
    }

private:
    AVPixelFormat openglPixelFormat = AV_PIX_FMT_RGB24;

    AVFormatContext* inputFormatContext = nullptr;
    AVCodecContext* codecContext = nullptr;

    int videoStreamIndex = -1;

    struct SwsContext* swsContext = nullptr;

    AVFrame* frameRGB = av_frame_alloc();
    uint8_t* buffer = nullptr;
    AVPacket* packet = av_packet_alloc();

    bool videoReady = false;

    std::thread videoReceiverThread;
    std::mutex frameRGBMutex;

    pose_id_t prevPoseID;

    void receiveVideo();

    int initFFMpeg();
    int initOutputFrame();
};

#endif // VIDEO_RECEIVER_H
