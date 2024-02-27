#ifndef VIDEOSTREAMER_H
#define VIDEOSTREAMER_H

#include <iostream>
#include <thread>

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

    float getFrameRate() {
        if (inputVideoStream != nullptr && inputVideoStream->avg_frame_rate.den > 0) {
            return inputVideoStream->avg_frame_rate.num / inputVideoStream->avg_frame_rate.den;
        }
        return 0;
    }

    int start(const std::string inputFileName, const std::string outputUrl);
    void cleanup();

    static VideoStreamer* create() {
        return new VideoStreamer();
    }

private:
    VideoStreamer() = default;
    ~VideoStreamer() = default;

    AVFormatContext* inputFormatContext = nullptr;
    AVFormatContext* outputFormatContext = nullptr;

    AVCodecContext* inputCodecContext = nullptr;
    AVCodecContext* outputCodecContext = nullptr;

    AVPacket inputPacket, outputPacket;

    int videoStreamIndex = -1;
    AVStream* inputVideoStream = nullptr;
    AVStream* outputVideoStream = nullptr;

    std::thread videoStreamerThread;

    void sendFrame();
};

#endif // VIDEOSTREAMER_H
