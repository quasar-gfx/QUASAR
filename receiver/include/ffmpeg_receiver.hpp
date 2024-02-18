#pragma once

#include <iostream>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
}

class FFmpegReceiver {
public:
    std::string inputUrl = "udp://localhost:1234"; // Specify the UDP input URL
    std::string outputFileName = "output.mp4";

    int frame_index;

    FFmpegReceiver() = default;
    ~FFmpegReceiver();
    int init();
    void receive();

private:
    AVFormatContext *inputContext = nullptr;
    AVFormatContext *outputContext = nullptr;

    AVCodecContext *codecContext = nullptr;

    AVPacket packet;
    int videoStreamIndex = -1;

    AVRational timeBase;
    AVRational streamTimeBase;
    AVStream *in_stream, *out_stream;
};
