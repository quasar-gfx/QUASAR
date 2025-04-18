#ifndef VIDEO_TEXTURE_H
#define VIDEO_TEXTURE_H

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/time.h>
#include <libavutil/imgutils.h>
}

#include <deque>
#include <atomic>
#include <thread>
#include <mutex>

#include <Utils/TimeUtils.h>

#include <Texture.h>

#include <CameraPose.h>

namespace quasar {

class VideoTexture : public Texture {
public:
    std::string videoURL = "127.0.0.1:12345";

    std::string formatName;

    struct Stats {
        float timeToReceiveMs = 0.0f;
        float timeToDecodeMs = 0.0f;
        float timeToResizeMs = 0.0f;
        float totalTimeToReceiveMs = 0.0f;
        float bitrateMbps = 0.0f;
    } stats;

    unsigned int videoWidth, videoHeight;

    VideoTexture(const TextureDataCreateParams &params,
                 const std::string &videoURL,
                 const std::string &formatName = "mpegts");
    ~VideoTexture();

    pose_id_t draw(pose_id_t poseID = -1);
    pose_id_t getLatestPoseID();

    void setMaxQueueSize(unsigned int maxQueueSize) {
        this->maxQueueSize = maxQueueSize;
    }

    float getFrameRate() {
        return 1.0f / timeutils::millisToSeconds(stats.totalTimeToReceiveMs);
    }

    void resize(unsigned int width, unsigned int height);

private:
    pose_id_t prevPoseID = -1;
    uint64_t framesReceived = 0;
    unsigned int maxQueueSize = 10;

    unsigned int internalWidth, internalHeight;

    int poseIDOffset = sizeof(pose_id_t) * 8;

    AVPixelFormat openglPixelFormat = AV_PIX_FMT_RGB24;

    AVFormatContext* inputFormatCtx = avformat_alloc_context();
    AVCodecContext* codecCtx = nullptr;

    int videoStreamIndex = -1;

    struct SwsContext* swsCtx = nullptr;

    AVFrame* frame = av_frame_alloc();
    AVPacket* packet = av_packet_alloc();

    std::atomic_bool videoReady = false;
    bool shouldTerminate = false;

    std::thread videoReceiverThread;
    std::mutex m;

    struct FrameData {
        pose_id_t poseID;
        AVFrame* frame;
        uint8_t* buffer;

        void free() {
            av_frame_free(&frame);
            delete[] buffer;
        }
    };

    std::deque<FrameData> frames;

    pose_id_t unpackPoseIDFromFrame(AVFrame* frame);

    void receiveVideo();

    int initFFMpeg();
};

} // namespace quasar

#endif // VIDEO_TEXTURE_H
