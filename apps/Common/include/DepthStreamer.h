#ifndef DEPTH_STREAMER_H
#define DEPTH_STREAMER_H

#include <thread>

#include <RenderTargets/RenderTarget.h>

#include <Networking/DataStreamerTCP.h>

#include <CameraPose.h>

#if !defined(__APPLE__) && !defined(__ANDROID__)
#include <cuda_gl_interop.h>
#include <Utils/CudaUtils.h>
#endif

class DepthStreamer : public RenderTarget {
public:
    std::string receiverURL;

    int imageSize;

    struct Stats {
        float timeToCopyFrameMs = -1.0f;
        float timeToSendMs = -1.0f;
        float bitrateMbps = -1.0f;
    } stats;

    DepthStreamer(const RenderTargetCreateParams &params, std::string receiverURL);
    ~DepthStreamer() {
        close();
    }

    void close();

    float getFrameRate() {
        return 1.0f / timeutils::millisToSeconds(stats.timeToSendMs);
    }

    void setTargetFrameRate(int targetFrameRate) {
        this->targetFrameRate = targetFrameRate;
    }

    void sendFrame(pose_id_t poseID);

private:
    int targetFrameRate = 60;

    DataStreamerTCP streamer;

    std::vector<uint8_t> data;
    RenderTarget* renderTargetCopy;

#if !defined(__APPLE__) && !defined(__ANDROID__)
    cudaGraphicsResource* cudaResource;

    struct CudaBuffer {
        pose_id_t poseID;
        cudaArray* buffer;
    };
    std::queue<CudaBuffer> cudaBufferQueue;

    std::thread dataSendingThread;
    std::mutex m;
    std::condition_variable cv;
    bool dataReady = false;

    std::atomic_bool running = false;

    void sendData();
#else
    pose_id_t poseID;
#endif
};

#endif // DEPTH_STREAMER_H
