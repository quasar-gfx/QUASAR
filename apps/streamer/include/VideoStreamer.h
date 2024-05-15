#ifndef VIDEOSTREAMER_H
#define VIDEOSTREAMER_H

#include <iostream>

#include <RenderTargets/RenderTarget.h>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/time.h>
#include <libavutil/opt.h>
}

#define MICROSECONDS_IN_SECOND 1e6f
#define MICROSECONDS_IN_MILLISECOND 1e3f

class VideoStreamer {
public:
    std::string outputUrl = "udp://localhost:1234";

    int targetFrameRate = 60;

    unsigned int framesSent = 0;

    struct Stats {
        float timeToEncode = -1.0f;
        float timeToCopyFrame = -1.0f;
        float timeToSendFrame = -1.0f;
        float totalTimeToSendFrame = -1.0f;
    } stats;

    explicit VideoStreamer() = default;
    ~VideoStreamer() = default;

    float getFrameRate() {
        return 1000.0f / stats.totalTimeToSendFrame;
    }

    int start(RenderTarget &renderTarget, const std::string outputUrl);
    void cleanup();

    void sendFrame(unsigned int poseId);

private:
    AVPixelFormat pixelFormat = AV_PIX_FMT_YUV420P;

    AVFormatContext* outputFormatContext = nullptr;
    AVCodecContext* outputCodecContext = nullptr;

    AVPacket outputPacket;

    int videoStreamIndex = -1;
    AVStream* outputVideoStream = nullptr;

    SwsContext* conversionContext;

    RenderTarget renderTarget;
    uint8_t* rgbaData;
    AVFrame* frame = av_frame_alloc();
};

#endif // VIDEOSTREAMER_H
