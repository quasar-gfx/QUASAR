#ifndef VIDEOSTREAMER_H
#define VIDEOSTREAMER_H

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/time.h>
#include <libavutil/opt.h>
}

#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <RenderTargets/RenderTarget.h>

#define MICROSECONDS_IN_SECOND 1e6f
#define MICROSECONDS_IN_MILLISECOND 1e3f

#define MBPS_TO_BPS 1e6f

class VideoStreamer {
public:
    std::string videoURL = "0.0.0.0:12345";

    unsigned int width, height;

    int targetFrameRate = 60;
    unsigned int targetBitRate = 50 * MBPS_TO_BPS;

    unsigned int framesSent = 0;

    struct Stats {
        float timeToEncode = -1.0f;
        float timeToCopyFrame = -1.0f;
        float timeToSendFrame = -1.0f;
        float totalTimeToSendFrame = -1.0f;
    } stats;

    explicit VideoStreamer(RenderTarget* renderTarget, const std::string &videoURL);
    ~VideoStreamer() = default;

    float getFrameRate() {
        return 1000.0f / stats.totalTimeToSendFrame;
    }

    void setTargetBitRate(unsigned int targetBitRate) {
        this->targetBitRate = targetBitRate;
        outputFormatCtx->bit_rate = targetBitRate;
    }

    void cleanup();

    void sendFrame(unsigned int poseID);

private:
    AVCodecID codecID = AV_CODEC_ID_H264;
    AVPixelFormat videoPixelFormat = AV_PIX_FMT_YUV420P;
    AVPixelFormat openglPixelFormat = AV_PIX_FMT_RGB24;

    AVFormatContext* outputFormatCtx = nullptr;
    AVCodecContext* codecCtx = nullptr;

    int videoStreamIndex = -1;
    AVStream* outputVideoStream = nullptr;

    SwsContext* conversionCtx;

    RenderTarget* renderTarget;
    uint8_t* rgbData;
    AVFrame* frame = av_frame_alloc();
    AVPacket* packet = av_packet_alloc();

    unsigned int poseID = -1;

    std::thread videoStreamerThread;
    std::mutex frameMutex;
    std::condition_variable cv;
    bool frameReady = false;

    void encodeAndSendFrame();
};

#endif // VIDEOSTREAMER_H
