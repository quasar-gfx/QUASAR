#ifndef RECORDER_H
#define RECORDER_H

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/time.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
}

#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <string>
#include <vector>
#include <filesystem>

#include <BS_thread_pool/BS_thread_pool.hpp>

#include <Path.h>
#include <RenderTargets/RenderTarget.h>
#include <Renderers/OpenGLRenderer.h>

#include <PostProcessing/PostProcessingEffect.h>

#if !defined(__APPLE__) && !defined(__ANDROID__)
#include <CudaGLInterop/CudaGLImage.h>
#endif

namespace quasar {

class Recorder : public RenderTarget {
public:
    enum class OutputFormat {
        MP4,
        PNG,
        JPG,
    };

    std::vector<std::string> outputFormats;

    int targetFrameRate;

    Recorder(
        const RenderTargetCreateParams &params,
        OpenGLRenderer& renderer,
        PostProcessingEffect& effect,
        const Path& outputPath,
        int targetFrameRate = 60,
        uint numThreads = 8)
            : RenderTarget(params)
            , renderer(renderer)
            , effect(effect)
            , targetFrameRate(targetFrameRate)
            , numThreads(numThreads)
            , outputPath(outputPath)
            , outputFormats({"MP4", "PNG", "JPG"})
#if !defined(__APPLE__) && !defined(__ANDROID__)
            , cudaImage(colorBuffer)
#endif
    {
        setOutputPath(outputPath);
    }
    ~Recorder();

    void saveScreenshotToFile(const Path &filename, bool saveAsHDR = false);

    void setOutputPath(const Path& path);
    void setFormat(OutputFormat format);
    void setTargetFrameRate(int targetFrameRate);

    void start();
    void stop();
    void captureFrame(const Camera &camera);

    const char* const* getFormatCStrArray() const {
        static std::vector<const char*> cstrs;
        cstrs.clear();
        for (const auto& fmt : outputFormats) {
            cstrs.push_back(fmt.c_str());
        }
        return cstrs.data();
    }

    int getFormatCount() const {
        return static_cast<int>(outputFormats.size());
    }

private:
    uint numThreads;

    OutputFormat outputFormat = OutputFormat::MP4;
    Path outputPath;

    OpenGLRenderer& renderer;
    PostProcessingEffect& effect;

    AVCodecID codecID = AV_CODEC_ID_H264;
    AVPixelFormat rgbaPixelFormat = AV_PIX_FMT_RGBA;
    AVPixelFormat videoPixelFormat = AV_PIX_FMT_YUV420P;

    struct FrameData {
        int ID;
        glm::vec3 position;
        glm::vec3 euler;
        std::vector<unsigned char> data;
        int64_t elapsedTime;
    };

    std::atomic<bool> running{false};
    std::atomic<int> frameCount{0};
    std::unique_ptr<BS::thread_pool<>> saveThreadPool;
    std::queue<FrameData> frameQueue;
    std::mutex queueMutex;
    std::mutex cameraPathMutex;
    std::mutex swsMutex;
    std::condition_variable queueCV;

    int64_t recordingStartTime;
    int64_t lastCaptureTime;
#if !defined(__APPLE__) && !defined(__ANDROID__)
    CudaGLImage cudaImage;
#endif

    AVFormatContext* outputFormatCtx = nullptr;
    AVCodecContext* codecCtx = nullptr;
    AVStream* outputVideoStream = nullptr;
    SwsContext* swsCtx = nullptr;

    void initializeFFmpeg();
    void finalizeFFmpeg();
    void saveFrames(int threadID);
};

} // namespace quasar

#endif // RECORDER_H
