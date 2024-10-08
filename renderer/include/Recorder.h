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
        : m_captureTarget()
        , m_running(false)
        , m_frameInterval(1.0 / fps)
        , m_outputPath(outputPath)
        , m_frameCount(0)
    {
        m_captureTarget = &renderer;
    }
    ~Recorder();

    void start();
    void stop();
    void captureFrame(GeometryBuffer& gbuffer);
    
private:
    void saveFrames();

    ForwardRenderer* m_captureTarget;
    std::thread m_saveThread;
    std::queue<std::vector<unsigned char>> m_frameQueue;
    std::mutex m_queueMutex;
    std::condition_variable m_queueCV;
    std::atomic<bool> m_running;
    std::chrono::duration<double> m_frameInterval;
    std::chrono::steady_clock::time_point m_lastCaptureTime;
    std::string m_outputPath;
    unsigned int m_frameCount;
};

#endif // RECORDER_H