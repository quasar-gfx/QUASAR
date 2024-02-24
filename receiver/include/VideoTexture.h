#ifndef VIDEO_RECEIVER_H
#define VIDEO_RECEIVER_H

#include <iostream>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}

#include "Texture.h"

class VideoTexture : public Texture {
public:
    std::string inputUrl = "udp://localhost:1234"; // Specify the UDP input URL

    int frameReceived;

    VideoTexture(unsigned int width, unsigned int height,
            GLenum format = GL_RGB, GLint wrap = GL_CLAMP_TO_EDGE, GLint filter = GL_LINEAR)
            : frameReceived(0), Texture(width, height, format, wrap, filter) { }

    ~VideoTexture() {
        cleanup();
    }

    int initVideo(const std::string inputUrl);
    int receive();
    void cleanup();

private:
    AVFormatContext* inputFormatContext = nullptr;

    AVCodecContext* inputCodecContext = nullptr;

    AVPacket packet;
    int videoStreamIndex = -1;

    AVFrame* frameRGB = nullptr;
    uint8_t* buffer = nullptr;
    struct SwsContext* swsContext = nullptr;

    int initFFMpeg();
    int initOutputFrame();
};

#endif // VIDEO_RECEIVER_H
