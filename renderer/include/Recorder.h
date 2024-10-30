#ifndef RECORDER_H
#define RECORDER_H

#include <Renderers/ForwardRenderer.h>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <string>


struct FrameData {
    std::vector<unsigned char> frame;
    glm::vec3 position;
    glm::vec3 euler;
};

class Recorder {
public:
    Recorder(float fps, const std::string& outputPath, ForwardRenderer &renderer)
        : captureTarget(&renderer)
        , running(false)
        , frameInterval(1.0 / fps)
        , outputPath(outputPath)
        , frameCount(0)
    {
        captureTarget = &renderer;
    }
    ~Recorder();
    void setOutputPath(const std::string& path);
    void start();
    void stop();
    void captureFrame(GeometryBuffer& gbuffer, Camera& camera);
    
private:
    void saveFrames();
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
};

#endif // RECORDER_H