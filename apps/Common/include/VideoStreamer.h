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

#include <RenderTargets/RenderTarget.h>

#define MICROSECONDS_IN_SECOND 1e6f
#define MICROSECONDS_IN_MILLISECOND 1e3f

class VideoStreamer {
public:
    std::string videoURL = "0.0.0.0:12345";

    unsigned int width, height;

    int targetFrameRate = 60;
    unsigned int targetBitRate = 100000 * 1000;

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
        outputFormatContext->bit_rate = targetBitRate;
    }

    void cleanup();

    void sendFrame(unsigned int poseId);

private:
    AVCodecID codecID = AV_CODEC_ID_H264;
    AVPixelFormat videoPixelFormat = AV_PIX_FMT_YUV420P;
    AVPixelFormat openglPixelFormat = AV_PIX_FMT_RGBA;

    AVFormatContext* outputFormatContext = nullptr;
    AVCodecContext* codecContext = nullptr;

    int videoStreamIndex = -1;
    AVStream* outputVideoStream = nullptr;

    SwsContext* conversionContext;

    RenderTarget* renderTarget;
    uint8_t* rgbaData;
    AVFrame* frame = av_frame_alloc();
    AVPacket* packet = av_packet_alloc();
};

#endif // VIDEOSTREAMER_H
