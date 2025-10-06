#ifndef BC4_DEPTH_STREAMER_H
#define BC4_DEPTH_STREAMER_H

#include <iomanip>
#include <thread>
#include <atomic>

#include <concurrentqueue/concurrentqueue.h>

#include <Shaders/ComputeShader.h>
#include <RenderTargets/RenderTarget.h>
#include <Networking/DataStreamerTCP.h>

#include <Path.h>
#include <Codec/BC4.h>
#include <Codec/ZSTDCodec.h>
#include <CameraPose.h>

#if defined(QUASAR_HAS_CUDA)
#include <CudaGLInterop/CudaGLBuffer.h>
#endif

namespace quasar {

class BC4DepthStreamer : public RenderTarget, public DataStreamerTCP {
public:
    uint width, height;

    Buffer bc4CompressedBuffer;

    std::string receiverURL;
    size_t compressedSize;

    struct Stats {
        double transferTimeMs = 0.0;
        double compressTimeMs = 0.0;
        double sendTimeMs = 0.0;
        double bitrateMbps = 0.0;
        double compressionRatio = 0.0;
    } stats;

    BC4DepthStreamer(const RenderTargetCreateParams& params, const std::string& receiverURL = "");
    ~BC4DepthStreamer();

    void stop();

    float getFrameRate() const {
        return 1.0f / timeutils::millisToSeconds(stats.sendTimeMs);
    }

    void setTargetFrameRate(int targetFrameRate) {
        this->targetFrameRate = targetFrameRate;
    }

    size_t compress(bool compress = false);
    void sendFrame(pose_id_t poseID);
    size_t writeToFile(const Path& filename);

private:
    int targetFrameRate = 30;

    std::vector<char> data;
    std::vector<char> compressedData;
    ZSTDCodec codec;

    ComputeShader bc4CompressionShader;

    size_t applyCodec();
    void writeToMemory(pose_id_t poseID = -1, void* cudaPtr = nullptr);

#if defined(QUASAR_HAS_CUDA)
    CudaGLBuffer cudaBufferBC4;

    struct CudaBuffer {
        pose_id_t poseID;
        void* buffer;
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

#endif // BC4_DEPTH_STREAMER_H
