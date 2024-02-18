#pragma once

#include <iostream>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}

class VideoReceiver {
public:
    std::string inputUrl = "udp://localhost:1234"; // Specify the UDP input URL
    std::string outputFileName = "output.mp4";

    int frameReceived;
    int textureWidth, textureHeight;

    unsigned int textureVideoBuffer;

    VideoReceiver() = default;
    ~VideoReceiver() = default;

    int init(int textureWidth, int textureHeight);
    void cleanup();
    void receive();

private:
    AVFormatContext* inputFormatContext = nullptr;

    AVCodecContext* inputCodecContext = nullptr;

    AVPacket packet;
    int videoStreamIndex = -1;

    AVFrame* frameRGB = nullptr;
    uint8_t* buffer = nullptr;
    struct SwsContext* swsContext = nullptr;

    int initFFMpeg();
    int initOutputTexture();
};
