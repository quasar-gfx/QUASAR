#ifndef VIDEOSTREAMER_H
#define VIDEOSTREAMER_H

#include <vector>
#include <atomic>
#include <thread>

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <concurrentqueue/concurrentqueue.h>

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

    struct Stats {
        double encodeTimeMs = 0.0;
        double transferTimeMs = 0.0;
        double sendTimeMs = 0.0;
        double totalSendTimeMs = 0.0;
    } stats;

    VideoStreamer(
        const RenderTargetCreateParams& params,
        const std::string& videoURL,
        int maxFrameRate = 30,
        int targetBitRateMbps = 12,
        bool useRTP = false);
    ~VideoStreamer();

    void stop();

    float getFrameRate() { return 1.0f / timeutils::millisToSeconds(stats.totalSendTimeMs); }

    void sendFrame(pose_id_t poseID);

private:
    uint64_t framesSent = 0;

    int maxFrameRate;
    int targetBitRateKbps;

#if defined(HAS_CUDA)
    std::string format = "NV12";
#else
    std::string format = "I420";
#endif
    std::string appSrcName = "oglsrc0";
    std::string payloaderName = "pay0";

    const int poseIDOffset = sizeof(pose_id_t) * 8;
    uint videoWidth, videoHeight;

#if defined(HAS_CUDA)
    CudaGLImage cudaGLImage;
#endif

    struct VideoFrame {
        pose_id_t poseID;
        std::vector<uint8_t> buffer;
    };
    moodycamel::ConcurrentQueue<VideoFrame> videoFrameQueue;

    std::thread videoStreamerThread;

    std::atomic_bool shouldTerminate = false;

    GstElement* pipeline = nullptr;
    GstElement* appsrc = nullptr;

    void encodeAndSendFrames();
    void packPoseIDIntoVideoFrame(pose_id_t poseID, uint8_t* data);
};

} // namespace quasar

#endif // VIDEOSTREAMER_H
