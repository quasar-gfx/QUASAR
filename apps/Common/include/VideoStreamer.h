#ifndef VIDEOSTREAMER_H
#define VIDEOSTREAMER_H

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>

#include <vector>
#include <queue>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <Utils/TimeUtils.h>
#include <RenderTargets/RenderTarget.h>
#include <CameraPose.h>

#if defined(HAS_CUDA)
#include <CudaGLInterop/CudaGLImage.h>
#endif

namespace quasar {

class VideoStreamer : public RenderTarget {
public:
    std::string videoURL = "0.0.0.0:12345";

    uint64_t framesSent = 0;

    struct Stats {
        double timeToEncodeMs = 0.0;
        double timeToTransferMs = 0.0;
        double timeToSendMs = 0.0;
        double totalTimeToSendMs = 0.0;
        double bitrateMbps = 0.0;
    } stats;

    VideoStreamer(
        const RenderTargetCreateParams& params,
        const std::string& videoURL,
        int targetFrameRate = 30,
        int targetBitRateMbps = 10);
    ~VideoStreamer();

    float getFrameRate();

    void setTargetFrameRate(int targetFrameRate);
    void setTargetBitRate(uint targetBitRate);

    void sendFrame(pose_id_t poseID);

private:
    int targetFrameRate;
    int targetBitRate;

    std::string format = "I420";
    std::string appSrcName = "oglsrc0";
    std::string payloaderName = "pay0";

    mutable std::atomic<uint64_t> totalBytesSent = 0;

    const int poseIDOffset = sizeof(pose_id_t) * 8;
    uint videoWidth, videoHeight;

    RenderTarget* renderTargetCopy;

#if defined(HAS_CUDA)
    CudaGLImage cudaGLImage;
    struct CudaBuffer {
        pose_id_t poseID;
        cudaArray_t buffer;
    };
    std::queue<CudaBuffer> cudaBufferQueue;
#else
    pose_id_t poseID = -1;
    std::vector<uint8_t> openglFrameData;
#endif

    std::vector<uint8_t> rgbaVideoFrameData;

    std::thread videoStreamerThread;
    std::mutex m;
    std::condition_variable cv;
    bool frameReady = false;

    std::atomic_bool videoReady = false;
    bool shouldTerminate = false;

    GstElement* pipeline = nullptr;
    GstElement* appsrc = nullptr;

    void encodeAndSendFrames();
    void packPoseIDIntoVideoFrame(pose_id_t poseID);
};

} // namespace quasar

#endif // VIDEOSTREAMER_H
