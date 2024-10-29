#ifndef VIDEOSTREAMER_H
#define VIDEOSTREAMER_H

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/time.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
#include <libavutil/avutil.h>
#if !defined(__APPLE__) && !defined(__ANDROID__)
#include <libavutil/hwcontext_cuda.h>
#endif
}

#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

#include <Utils/TimeUtils.h>
#include <RenderTargets/RenderTarget.h>
#include <CameraPose.h>

#if !defined(__APPLE__) && !defined(__ANDROID__)
#include <cuda_gl_interop.h>
#include <Utils/CudaUtils.h>
#endif

class VideoStreamer : public RenderTarget {
public:
    std::string videoURL = "0.0.0.0:12345";

    std::string formatName;

    uint64_t framesSent = 0;

    struct Stats {
        float timeToEncodeMs = -1.0f;
        float timeToCopyFrameMs = -1.0f;
        float timeToSendMs = -1.0f;
        float totalTimeToSendMs = -1.0f;
        float bitrateMbps = -1.0f;
    } stats;

    VideoStreamer(const RenderTargetCreateParams &params,
                  const std::string &videoURL,
                  unsigned int targetBitRateMbps = 50,
                  const std::string &formatName = "mpegts");
    ~VideoStreamer();

    float getFrameRate() {
        return 1.0f / timeutils::millisToSeconds(stats.totalTimeToSendMs);
    }

    void setTargetFrameRate(int targetFrameRate) {
        this->targetFrameRate = targetFrameRate;
        codecCtx->time_base = {1, targetFrameRate};
        codecCtx->framerate = {targetFrameRate, 1};
    }

    void setTargetBitRate(unsigned int targetBitRate) {
        this->targetBitRate = targetBitRate;
        outputFormatCtx->bit_rate = targetBitRate;
    }

    void sendFrame(pose_id_t poseID);

private:
    int targetFrameRate = 60;
    unsigned int targetBitRate;

    int poseIDOffset = sizeof(pose_id_t) * 8;

    unsigned int videoWidth, videoHeight;

    RenderTarget* renderTargetCopy;

    AVCodecID codecID = AV_CODEC_ID_H264;
    AVPixelFormat rgbaPixelFormat = AV_PIX_FMT_RGBA;
    AVPixelFormat videoPixelFormat = AV_PIX_FMT_YUV420P;

    AVFormatContext* outputFormatCtx = nullptr;
    AVCodecContext* codecCtx = nullptr;

    int videoStreamIndex = 0;
    AVStream* outputVideoStream = nullptr;

    SwsContext* swsCtx = nullptr;

    void packPoseIDIntoVideoFrame(pose_id_t poseID);

#if !defined(__APPLE__) && !defined(__ANDROID__)
    cudaGraphicsResource* cudaResource;

    struct CudaBuffer {
        pose_id_t poseID;
        cudaArray* buffer;
    };
    std::queue<CudaBuffer> cudaBufferQueue;
#else
    pose_id_t poseID = -1;

    std::vector<uint8_t> openglFrameData;
#endif

    std::vector<uint8_t> rgbaVideoFrameData;
    AVFrame* frame = av_frame_alloc();
    AVPacket* packet = av_packet_alloc();

    std::thread videoStreamerThread;
    std::mutex m;
    std::condition_variable cv;
    bool frameReady = false;

    std::atomic_bool sendFrames = false;
    bool shouldTerminate = false;

#if !defined(__APPLE__) && !defined(__ANDROID__)
    int initCuda();
#endif
    void encodeAndSendFrames();
};

#endif // VIDEOSTREAMER_H
