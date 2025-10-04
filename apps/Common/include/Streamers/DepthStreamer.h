#ifndef DEPTH_STREAMER_H
#define DEPTH_STREAMER_H

#include <thread>
#include <atomic>
#include <vector>
#include <memory>

#include <CameraPose.h>
#include <RenderTargets/RenderTarget.h>
#include <Networking/DataStreamerTCP.h>
#include <concurrentqueue/concurrentqueue.h>

#if defined(QUASAR_HAS_CUDA)
#include <CudaGLInterop/CudaGLImage.h>
#endif

namespace quasar {

class DepthStreamer : public RenderTarget, public DataStreamerTCP {
public:
    std::string receiverURL;

    int imageSize;

    struct Stats {
        double transferTimeMs = 0.0;
        double sendTimeMs = 0.0;
        double bitrateMbps = 0.0;
    } stats;

    DepthStreamer(const RenderTargetCreateParams& params, std::string receiverURL);
    ~DepthStreamer();

    void stop();

    float getFrameRate() {
        return 1.0f / timeutils::millisToSeconds(stats.sendTimeMs);
    }

    void setTargetFrameRate(int targetFrameRate) {
        this->targetFrameRate = targetFrameRate;
    }

    void sendFrame(pose_id_t poseID);

private:
    int targetFrameRate = 30;

    std::vector<char> data;
    RenderTarget renderTargetCopy;

#if defined(QUASAR_HAS_CUDA)
    CudaGLImage cudaGLImage;

    struct CudaBuffer {
        pose_id_t poseID;
        cudaArray_t buffer;
    };
    moodycamel::ConcurrentQueue<CudaBuffer> cudaBufferQueue;

    std::atomic_bool running{false};
    std::thread dataSendingThread;

    void sendData();
#else
    pose_id_t poseID;
#endif
};

} // namespace quasar

#endif // DEPTH_STREAMER_H
