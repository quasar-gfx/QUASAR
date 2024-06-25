#ifndef DEPTH_STREAMER_H
#define DEPTH_STREAMER_H

#include <thread>

#include <RenderTargets/RenderTarget.h>

#include <Networking/DataStreamerTCP.h>

#include <CameraPose.h>

#ifndef __APPLE__
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

    explicit DepthStreamer(const RenderTargetCreateParams &params, std::string receiverURL);
    ~DepthStreamer() {
        close();
    }

    void close();

    void sendFrame(pose_id_t poseID);

private:
    DataStreamerTCP streamer;

    std::vector<uint8_t> data;
    RenderTarget* renderTargetCopy;

#ifdef __APPLE__
    pose_id_t poseID;
#else
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
#endif
};

#endif // DEPTH_STREAMER_H
