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

class VideoTexture : public Texture {
public:
    std::string videoURL = "localhost:1234";

    unsigned int width, height;

    int targetFrameRate = 60;

    int frameReceived = 0;

    struct Stats {
        float timeToReceiveFrame = -1.0f;
        float timeToDecode = -1.0f;
        float timeToResize = -1.0f;
        float totalTimeToReceiveFrame = -1.0f;
    } stats;

    explicit VideoTexture(const TextureCreateParams &params) : Texture(params), width(params.width), height(params.height) { }

    ~VideoTexture() {
        cleanup();
    }

    void initVideo(const std::string &videoURL);
    void cleanup();

    pose_id_t draw();

    float getFrameRate() {
        return 1.0f / stats.totalTimeToReceiveFrame;
    }

private:
    AVCodecID codecID = AV_CODEC_ID_H264;
    AVPixelFormat pixelFormat = AV_PIX_FMT_YUV420P;

    AVFormatContext* inputFormatContext = nullptr;
    AVCodecContext* codecContext = nullptr;

    struct SwsContext* swsContext = nullptr;

    AVFrame* frameRGB = nullptr;
    uint8_t* buffer = nullptr;
    AVPacket* packet = av_packet_alloc();

    bool videoReady = false;

    std::thread videoReceiverThread;
    std::mutex frameRGBMutex;

    void receiveVideo();

    int initFFMpeg();
    int initOutputFrame();
};

#endif // VIDEO_RECEIVER_H
