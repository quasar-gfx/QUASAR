#ifndef RECORDER_H
#define RECORDER_H

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/time.h>
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

#include <RenderTargets/RenderTarget.h>
#include <RenderTargets/GBuffer.h>
#include <Renderers/OpenGLRenderer.h>

class Recorder {
public:
    struct FrameData {
        std::vector<unsigned char> frame;
        glm::vec3 position;
        glm::vec3 euler;
        int64_t pts;
    };

    enum class OutputFormat {
        PNG,
        JPG,
        MP4
    };

    Recorder(OpenGLRenderer &renderer, const std::string& outputPath, float targetFrameRate)
        : captureTarget(&renderer)
        , running(false)
        , frameInterval(1.0 / targetFrameRate)
        , outputPath(outputPath)
        , frameCount(0)
        , outputFormat(OutputFormat::PNG) { }
    ~Recorder();

    void setOutputPath(const std::string& path);
    void setTargetFrameRate(int targetFrameRate);
    void start();
    void stop();
    void captureFrame(GeometryBuffer& gbuffer, Camera& camera);
    void setFormat(OutputFormat format);
    void updateResolution(int width, int height);

private:
    static const int NUM_SAVE_THREADS = 16;
    OutputFormat outputFormat = OutputFormat::PNG;

    std::chrono::steady_clock::time_point recordingStartTime;

    std::vector<std::thread> saveThreadPool;
    std::atomic<bool> running{false};
    std::atomic<size_t> frameCount{0};
    OpenGLRenderer* captureTarget;
    std::thread saveThread;
    std::queue<FrameData> frameQueue;
    std::mutex queueMutex;
    std::condition_variable queueCV;
    std::chrono::duration<double> frameInterval;
    std::chrono::steady_clock::time_point lastCaptureTime;
    std::string outputPath;

    AVFormatContext* formatContext = nullptr;
    AVCodecContext* codecContext = nullptr;
    AVStream* videoStream = nullptr;
    SwsContext* swsContext = nullptr;
    AVFrame* frame = nullptr;
    int frameIndex = 0;

    void initializeFFmpeg();
    void finalizeFFmpeg();
    void saveFrames();
};

#endif // RECORDER_H
