#ifndef BC4_DEPTH_STREAMER_H
#define BC4_DEPTH_STREAMER_H

#include <iostream>
#include <iomanip>
#include <thread>
#include <queue>
#include <atomic>
#include <RenderTargets/RenderTarget.h>
#include <Networking/DataStreamerTCP.h>
#include <CameraPose.h>
#include <glm/glm.hpp>

#include <Shaders/ComputeShader.h>
#include <lz4_stream/lz4_stream.h>

#if !defined(__APPLE__) && !defined(__ANDROID__)
#include <cuda_gl_interop.h>
#include <Utils/CudaUtils.h>
#endif

class BC4DepthStreamer : public RenderTarget {
public:
    struct Block {
        float max;
        float min;
        uint32_t data[6];
    };
    Buffer<Block> bc4CompressedBuffer;

    std::string receiverURL;
    unsigned int compressedSize;

    // struct Stats {
    //     float timeToCopyFrameMs = -1.0f;
    //     float timeToSendMs = -1.0f;
    //     float bitrateMbps = -1.0f;
    // } stats;

    struct StreamerStats {
        float timeToCompressMs = -1.0f;
        float timeToCopyFrameMs = -1.0f;
        float timeToSendMs = -1.0f;
        float bitrateMbps = -1.0f;
        float lz4CompressionRatio = -1.0f;
    };

    StreamerStats stats;

    BC4DepthStreamer(const RenderTargetCreateParams &params, std::string receiverURL);
    ~BC4DepthStreamer();

    void close();

    float getFrameRate() const {
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
    
    // lz4
    std::vector<uint8_t> lz4Buffer;

    // BC4 compute shader
    ComputeShader bc4CompressionShader;
    void compressBC4();

#if !defined(__APPLE__) && !defined(__ANDROID__)
    cudaGraphicsResource* cudaResource;

    struct CudaBuffer {
        pose_id_t poseID;
        void* buffer;
    };
    std::queue<CudaBuffer> cudaBufferQueue;

    std::thread dataSendingThread;
    std::mutex m;
    std::condition_variable cv;
    bool dataReady = false;

    std::atomic_bool running{false};

    void sendData();
#else
    pose_id_t poseID;
#endif
};

#endif // BC4_DEPTH_STREAMER_H
