#ifndef VIDEO_RECEIVER_H
#define VIDEO_RECEIVER_H

#include <iostream>
#include <thread>
#include <mutex>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/time.h>
#include <libavutil/imgutils.h>
}

#include <Texture.h>

class VideoTexture : public Texture {
public:
    std::string inputUrl = "udp://localhost:1234";

    unsigned int width, height;

    int frameReceived = 0;
    float totalTimeToReceiveFrame = 0.0f;

    explicit VideoTexture(const TextureCreateParams &params) : Texture(params), width(params.width), height(params.height) { }

    ~VideoTexture() {
        cleanup();
    }

    void initVideo(const std::string &inputUrl);
    void cleanup();

    unsigned int draw();

    float getFrameRate() {
        return 1.0f / totalTimeToReceiveFrame;
    }

private:
    AVFormatContext* inputFormatContext = nullptr;
    AVCodecContext* inputCodecContext = nullptr;

    AVPacket packet;
    int videoStreamIndex = -1;

    AVFrame* frameRGB = nullptr;
    uint8_t* buffer = nullptr;
    struct SwsContext* swsContext = nullptr;

    bool videoReady = false;

    std::thread videoReceiverThread;
    std::mutex frameRGBMutex;

    void receiveVideo();

    int initFFMpeg();
    int initOutputFrame();
};

#endif // VIDEO_RECEIVER_H
