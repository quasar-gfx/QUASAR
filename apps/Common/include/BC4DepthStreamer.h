#ifndef BC4_DEPTH_STREAMER_H
#define BC4_DEPTH_STREAMER_H

#include <iomanip>
#include <thread>
#include <queue>
#include <atomic>

#include <glm/glm.hpp>

#include <RenderTargets/RenderTarget.h>
#include <Networking/DataStreamerTCP.h>
#include <CameraPose.h>

#include <Shaders/ComputeShader.h>
#include <Codec/ZSTDCodec.h>

#if !defined(__APPLE__) && !defined(__ANDROID__)
#include <CudaGLInterop/CudaGLBuffer.h>
#endif

namespace quasar {

class BC4DepthStreamer : public RenderTarget {
public:
    struct Block {
        float max;
        float min;
        uint32_t data[6];
    };
    Buffer bc4CompressedBuffer;

    std::string receiverURL;
    unsigned int compressedSize;

    struct Stats {
        double timeToCopyFrameMs = 0.0f;
        double timeToCompressMs = 0.0f;
        double timeToSendMs = 0.0f;
        double bitrateMbps = 0.0f;
        double compressionRatio = 0.0f;
    } stats;

    BC4DepthStreamer(const RenderTargetCreateParams &params, const std::string &receiverURL = "");
    ~BC4DepthStreamer();

    void close();

    float getFrameRate() const {
        return 1.0f / timeutils::millisToSeconds(stats.timeToSendMs);
    }

    void setTargetFrameRate(int targetFrameRate) {
        this->targetFrameRate = targetFrameRate;
    }

    unsigned int compress(bool compress = false);
    void sendFrame(pose_id_t poseID);

private:
    int targetFrameRate = 30;
    DataStreamerTCP streamer;

    std::vector<char> data;
    std::vector<char> compressedData;
    ZSTDCodec codec;

    ComputeShader bc4CompressionShader;

    unsigned int applyCodec();
    void copyFrameToCPU(pose_id_t poseID = -1, void* cudaPtr = nullptr);

#if !defined(__APPLE__) && !defined(__ANDROID__)
    CudaGLBuffer cudaBufferBc4;

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

} // namespace quasar

#endif // BC4_DEPTH_STREAMER_H
