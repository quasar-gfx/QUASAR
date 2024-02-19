#include <iostream>

#include "video_streamer.hpp"

int main(int argc, char **argv) {
    std::string inputFileName = "input.mp4";
    std::string outputUrl = "udp://localhost:1234";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-i") && i + 1 < argc) {
            inputFileName = argv[i + 1];
            i++;
        }
        else if (!strcmp(argv[i], "-o") && i + 1 < argc) {
            outputUrl = argv[i + 1];
            i++;
        }
    }

    VideoStreamer* streamer = new VideoStreamer();
    int ret = streamer->init(inputFileName, outputUrl);
    if (ret < 0) {
        std::cerr << "Failed to initialize FFMpeg Video Streamer" << std::endl;
        return ret;
    }

    while (1) {
        ret = streamer->sendFrame();
        if (ret < 0) {
            std::cerr << "Failed to send frame" << std::endl;
            break;
        }
    }

    std::cout << "Sent " << streamer->framesSent << " frames" << std::endl;

    streamer->cleanup();

    return 0;
}
