#ifndef RECORDER_H
#define RECORDER_H

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/time.h>
#include <libavutil/imgutils.h>
}

#include <Renderers/ForwardRenderer.h>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <string>
#include <vector>

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


class Recorder {
public:
    Recorder(float fps, const std::string& outputPath, ForwardRenderer &renderer)
        : captureTarget(&renderer)
        , running(false)
        , frameInterval(1.0 / fps)
        , outputPath(outputPath)
        , frameCount(0)
        , outputFormat(OutputFormat::PNG)
    {
    }
    ~Recorder();
    void setOutputPath(const std::string& path);
    void setFrameRate(int fps);
    void start();
    void stop();
    void captureFrame(GeometryBuffer& gbuffer, Camera& camera);
    void setOutputFormat(OutputFormat format);
    
private:
    void saveFrames();
    std::chrono::steady_clock::time_point recordingStartTime;
    OutputFormat outputFormat = OutputFormat::PNG;
    static const int NUM_SAVE_THREADS = 16;
    std::vector<std::thread> saveThreadPool;
    std::atomic<bool> running{false};
    std::atomic<size_t> frameCount{0};
    ForwardRenderer* captureTarget;
    std::thread saveThread;
    std::queue<FrameData> frameQueue;
    std::mutex queueMutex;
    std::condition_variable queueCV;
    std::chrono::duration<double> frameInterval;
    std::chrono::steady_clock::time_point lastCaptureTime;
    std::string outputPath;
    void initializeFFmpeg();
    void finalizeFFmpeg();
    AVFormatContext* formatContext = nullptr;
    AVCodecContext* codecContext = nullptr;
    AVStream* videoStream = nullptr;
    SwsContext* swsContext = nullptr;
    AVFrame* frame = nullptr;
    int frameIndex = 0;
};

#endif // RECORDER_H