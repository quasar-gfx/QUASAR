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

    Recorder(OpenGLRenderer &renderer, Shader &shader, const std::string& outputPath, float targetFrameRate = 30)
        : renderer(renderer)
        , shader(shader)
        , renderTargetTemp({
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
        , frameInterval(1.0 / targetFrameRate)
        , outputPath(outputPath)
        , outputFormat(OutputFormat::PNG) { }
    Recorder(OpenGLRenderer &renderer, Shader &shader, float targetFrameRate = 30)
        : Recorder(renderer, shader, ".", targetFrameRate) { }
    ~Recorder();

    void saveScreenshotToFile(const std::string &filename, bool saveAsHDR = false);

    void setOutputPath(const std::string& path);
    void setFormat(OutputFormat format);
    void setTargetFrameRate(int targetFrameRate);

    void start();
    void stop();
    void captureFrame(Camera& camera);

private:
    static const int NUM_SAVE_THREADS = 16;
    OutputFormat outputFormat = OutputFormat::PNG;
    std::string outputPath;

    OpenGLRenderer& renderer;
    Shader& shader;

    RenderTarget renderTargetTemp;

    std::atomic<bool> running{false};
    std::atomic<size_t> frameCount{0};
    std::vector<std::thread> saveThreadPool;
    std::queue<FrameData> frameQueue;
    std::mutex queueMutex;
    std::condition_variable queueCV;

    std::chrono::duration<double> frameInterval;
    std::chrono::steady_clock::time_point recordingStartTime;
    std::chrono::steady_clock::time_point lastCaptureTime;

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
