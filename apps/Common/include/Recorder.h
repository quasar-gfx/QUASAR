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

#include <RenderTargets/RenderTarget.h>
#include <Renderers/OpenGLRenderer.h>

#include <PostProcessing/PostProcessingEffect.h>

#if !defined(__APPLE__) && !defined(__ANDROID__)
#include <CudaGLInterop/CudaGLImage.h>
#endif

class Recorder {
public:
    enum class OutputFormat {
        MP4,
        PNG,
        JPG,
    };

    int targetFrameRate;

    Recorder(OpenGLRenderer &renderer, PostProcessingEffect &effect, const std::string& outputPath, int targetFrameRate = 60, unsigned int numThreads = 8)
            : renderer(renderer)
            , effect(effect)
            , renderTargetCopy({
                .width = renderer.width,
                .height = renderer.height,
                .internalFormat = GL_RGBA,
                .format = GL_RGBA,
                .type = GL_UNSIGNED_BYTE,
                .wrapS = GL_CLAMP_TO_EDGE,
                .wrapT = GL_CLAMP_TO_EDGE,
                .minFilter = GL_LINEAR,
                .magFilter = GL_LINEAR
            })
            , targetFrameRate(targetFrameRate)
            , numThreads(numThreads)
            , outputPath(outputPath)
#if !defined(__APPLE__) && !defined(__ANDROID__)
            , cudaImage(renderTargetCopy.colorBuffer)
#endif
            {
    }
    Recorder(OpenGLRenderer &renderer, PostProcessingEffect &effect, int targetFrameRate = 60, unsigned int numThreads = 8)
        : Recorder(renderer, effect, ".", targetFrameRate) { }
    ~Recorder();

    void saveScreenshotToFile(const std::string &filename, bool saveAsHDR = false);

    void setOutputPath(const std::string& path);
    void setFormat(OutputFormat format);
    void setTargetFrameRate(int targetFrameRate);

    void start();
    void stop();
    void captureFrame(const Camera &camera);

private:
    unsigned int numThreads;

    OutputFormat outputFormat = OutputFormat::MP4;
    std::string outputPath;

    OpenGLRenderer& renderer;
    PostProcessingEffect& effect;

    RenderTarget renderTargetCopy;

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
    std::unique_ptr<BS::thread_pool> saveThreadPool;
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

#endif // RECORDER_H
