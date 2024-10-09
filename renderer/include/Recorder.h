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

    void start();
    void stop();
    void captureFrame(GeometryBuffer& gbuffer);
    
private:
    void saveFrames();

    ForwardRenderer* captureTarget;
    std::thread saveThread;
    std::queue<std::vector<unsigned char>> frameQueue;
    std::mutex queueMutex;
    std::condition_variable queueCV;
    std::atomic<bool> running;
    std::chrono::duration<double> frameInterval;
    std::chrono::steady_clock::time_point lastCaptureTime;
    std::string outputPath;
    unsigned int frameCount;
};

#endif // RECORDER_H