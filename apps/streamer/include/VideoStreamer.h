#ifndef VIDEOSTREAMER_H
#define VIDEOSTREAMER_H

#include <iostream>

#include <Texture.h>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/time.h>
#include <libavutil/opt.h>
}

class VideoStreamer {
public:
    std::string outputUrl = "udp://localhost:1234";

    int frameRate = 15;

    unsigned int framesSent = 0;

    float getFrameRate() {
        return 0;
    }

    int start(Texture* texture, const std::string outputUrl);
    void cleanup();

    void sendFrame();

    static VideoStreamer* create() {
        return new VideoStreamer();
    }

private:
    VideoStreamer() = default;
    ~VideoStreamer() = default;

#ifdef __APPLE__
    AVPixelFormat pixelFormat = AV_PIX_FMT_YUV420P;
#else
    AVPixelFormat pixelFormat = AV_PIX_FMT_YUV444P;
#endif

    AVFormatContext* outputFormatContext = nullptr;
    AVCodecContext* outputCodecContext = nullptr;

    AVPacket outputPacket;

    int videoStreamIndex = -1;
    AVStream* outputVideoStream = nullptr;

    SwsContext* conversionContext;

    Texture* sourceTexture;
    uint8_t* rgbaData;
};

#endif // VIDEOSTREAMER_H
