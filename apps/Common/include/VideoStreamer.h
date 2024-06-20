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
#ifndef __APPLE__
#include <libavutil/hwcontext_cuda.h>
#endif
}

#include <iostream>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <RenderTargets/RenderTarget.h>

#include <CameraPose.h>

#ifndef __APPLE__
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#endif

#define MICROSECONDS_IN_SECOND 1e6f
#define MICROSECONDS_IN_MILLISECOND 1e3f

#define MBPS_TO_BPS 1e6f

class VideoStreamer {
public:
    std::string videoURL = "0.0.0.0:12345";

    unsigned int width, height;

    unsigned int framesSent = 0;

    struct Stats {
        float timeToEncode = -1.0f;
        float timeToCopyFrame = -1.0f;
        float timeToSendFrame = -1.0f;
        float totalTimeToSendFrame = -1.0f;
    } stats;

    explicit VideoStreamer(RenderTarget* renderTarget, const std::string &videoURL);
    ~VideoStreamer() {
        cleanup();
    }

    float getFrameRate() {
        return 1000.0f / stats.totalTimeToSendFrame;
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

    void cleanup();

    void sendFrame(pose_id_t poseID);

private:
    int targetFrameRate = 60;
    unsigned int targetBitRate = 50 * MBPS_TO_BPS;

    pose_id_t poseID = -1;

    AVCodecID codecID = AV_CODEC_ID_H264;
#ifndef __APPLE__
    AVPixelFormat bufferPixelFormat = AV_PIX_FMT_RGBA;
#else
    AVPixelFormat bufferPixelFormat = AV_PIX_FMT_RGB24;
#endif
    AVPixelFormat videoPixelFormat = AV_PIX_FMT_YUV420P;

    AVFormatContext* outputFormatCtx = nullptr;
    AVCodecContext* codecCtx = nullptr;

    int videoStreamIndex = -1;
    AVStream* outputVideoStream = nullptr;

    SwsContext* swsCtx = nullptr;

#ifndef __APPLE__
    cudaGraphicsResource* cudaResource;
    cudaArray* cudaBuffer;

    AVBufferRef* deviceCtx = nullptr;
    AVBufferRef* cudaDeviceCtx = nullptr;
    AVBufferRef* frameCtx = nullptr;
    AVBufferRef* cudaFrameCtx = nullptr;
#endif

    RenderTarget* renderTarget;
    uint8_t* rgbData;
    AVFrame* frame = av_frame_alloc();
    AVPacket* packet = av_packet_alloc();

    std::thread videoStreamerThread;
    std::mutex frameMutex;
    std::condition_variable cv;
    bool frameReady = false;

    std::atomic_bool sendFrames = false;
    bool shouldTerminate = false;

#ifndef __APPLE__
    CUdevice findCudaDevice();
    int initCuda();
#endif
    void encodeAndSendFrames();
};

#endif // VIDEOSTREAMER_H
