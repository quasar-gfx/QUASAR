#pragma once

#include <iostream>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/time.h>
#include <libavutil/opt.h>
}

class VideoStreamer {
public:
    std::string inputFileName = "input.mp4";
    std::string outputUrl = "udp://localhost:1234";

    unsigned int framesSent;

    VideoStreamer() = default;
    ~VideoStreamer() = default;

    int init(const std::string inputFileName, const std::string outputUrl);
    void cleanup();

    int sendFrame();

private:
    AVFormatContext *inputFormatContext = nullptr;
    AVFormatContext *outputFormatContext = nullptr;

    AVCodecContext *inputCodecContext = nullptr;
    AVCodecContext *outputCodecContext = nullptr;

    AVPacket inputPacket, outputPacket;

    int videoStreamIndex = -1;
    AVStream *inputVideoStream = nullptr;
    AVStream *outputVideoStream = nullptr;
};

