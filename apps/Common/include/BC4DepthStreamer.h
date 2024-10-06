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

#if !defined(__APPLE__) && !defined(__ANDROID__)
#include <cuda_gl_interop.h>
#include <Utils/CudaUtils.h>
#endif

struct Block {
    float max;
    float min;
    uint32_t arr[6];
};

class BC4DepthStreamer : public RenderTarget {
public:
    std::string receiverURL;
    unsigned int compressedSize;

    struct Stats {
        float timeToCopyFrameMs = -1.0f;
        float timeToSendMs = -1.0f;
        float bitrateMbps = -1.0f;
    } stats;

    BC4DepthStreamer(const RenderTargetCreateParams &params, std::string receiverURL);
    ~BC4DepthStreamer();
    void close();
    float getFrameRate() const {
        return 1.0f / timeutils::millisToSeconds(stats.timeToSendMs);
    }
    void setTargetFrameRate(int targetFrameRate) {
        this->targetFrameRate = targetFrameRate;
    }
    //void sendFrame(pose_id_t poseID);
    void sendFrame(pose_id_t poseID, const Texture& depthStencilBuffer, ComputeShader& bc4CompressShader, const glm::uvec2& windowSize);

    GLuint getBC4Buffer() const { return bc4Buffer; }

private:
    int targetFrameRate = 60;
    DataStreamerTCP streamer;

    void compressBC4(const Texture& depthStencilBuffer, ComputeShader& bc4CompressShader, const glm::uvec2& windowSize);

    //bc4
    std::vector<uint8_t> compressedData;
    GLuint bc4Buffer;

    void debugPrintData(const std::vector<uint8_t>& data, size_t bytesToPrint = 64) {
        std::cout << "Sending data (first " << bytesToPrint << " bytes):" << std::endl;
        for (size_t i = 0; i < std::min(data.size(), bytesToPrint); ++i) {
            std::cout << std::setw(2) << std::setfill('0') << std::hex << static_cast<int>(data[i]) << " ";
            if ((i + 1) % 16 == 0) std::cout << std::endl;
        }
        std::cout << std::dec << std::endl;
    }

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