#ifndef VIDEO_RECEIVER_H
#define VIDEO_RECEIVER_H

#include <iostream>
#include <thread>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}

#include <Texture.h>

class VideoTexture : public Texture {
public:
    std::string inputUrl = "udp://localhost:1234"; // Specify the UDP input URL

    int frameReceived;

    void initVideo(const std::string inputUrl);
    void cleanup();

    void draw();

    float getFrameRate() {
        if (inputFormatContext == nullptr || videoStreamIndex == -1) {
            return 0;
        }
        return av_q2d(inputFormatContext->streams[videoStreamIndex]->avg_frame_rate);
    }

    static VideoTexture* create(unsigned int width, unsigned int height,
            GLenum format = GL_RGB,
            GLint wrapS = GL_CLAMP_TO_EDGE, GLint wrapT = GL_CLAMP_TO_EDGE,
            GLint minFilter = GL_LINEAR, GLint magFilter = GL_LINEAR) {
        return new VideoTexture(width, height, format, wrapS, wrapT, minFilter, magFilter);
    }

private:
    VideoTexture(unsigned int width, unsigned int height,
            GLenum format = GL_RGB,
            GLint wrapS = GL_CLAMP_TO_EDGE, GLint wrapT = GL_CLAMP_TO_EDGE,
            GLint minFilter = GL_LINEAR, GLint magFilter = GL_LINEAR)
                : frameReceived(0), Texture(width, height, TEXTURE_DIFFUSE, format, wrapS, wrapT, minFilter, magFilter) { }

    ~VideoTexture() {
        cleanup();
    }

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
