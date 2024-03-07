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

    int frameReceived = 0;
    float timeToReceiveFrame = 0.0f;

    VideoTexture(unsigned int width, unsigned int height,
            GLint internalFormat = GL_RGB,
            GLenum format = GL_RGB,
            GLenum type = GL_UNSIGNED_BYTE,
            GLint wrapS = GL_CLAMP_TO_EDGE, GLint wrapT = GL_CLAMP_TO_EDGE,
            GLint minFilter = GL_LINEAR, GLint magFilter = GL_LINEAR) :
            Texture(width, height, internalFormat, format, type, wrapS, wrapT, minFilter, magFilter) {
    }

    ~VideoTexture() {
        cleanup();
    }

    void initVideo(const std::string &inputUrl);
    void cleanup();

    void draw();

    float getFrameRate() {
        return 1.0f / timeToReceiveFrame;
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
