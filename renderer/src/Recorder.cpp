#include <Recorder.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include "Utils/FileIO.h"

Recorder::~Recorder() {
    stop();
}

void Recorder::start() {
    m_running = true;
    m_lastCaptureTime = std::chrono::steady_clock::now();
    m_saveThread = std::thread(&Recorder::saveFrames, this);
}

void Recorder::stop() {
    m_running = false;
    m_queueCV.notify_one();
    if (m_saveThread.joinable()) {
        m_saveThread.join();
    }
}

void Recorder::captureFrame(GeometryBuffer& gbuffer) {
    auto currentTime = std::chrono::steady_clock::now();
    if (currentTime - m_lastCaptureTime >= m_frameInterval) {
        std::vector<unsigned char> frameData(m_captureTarget->width * m_captureTarget->height * 4);
        
        // Bind the source RenderTarget
        gbuffer.colorBuffer.bind(0);
        // Read pixels directly from the source RenderTarget
        glReadPixels(0, 0, m_captureTarget->width, m_captureTarget->height, GL_RGBA, GL_UNSIGNED_BYTE, frameData.data());
        // Unbind the source RenderTarget
        gbuffer.colorBuffer.unbind();

        {
            std::lock_guard<std::mutex> lock(m_queueMutex);
            m_frameQueue.push(std::move(frameData));
        }
        m_queueCV.notify_one();

        m_lastCaptureTime = currentTime;
    }
}

void Recorder::saveFrames() {
    while (m_running || !m_frameQueue.empty()) {
        std::vector<unsigned char> frameData;
        {
            std::unique_lock<std::mutex> lock(m_queueMutex);
            m_queueCV.wait(lock, [this] { return !m_frameQueue.empty() || !m_running; });
            if (!m_running && m_frameQueue.empty()) {
                break;
            }
            frameData = std::move(m_frameQueue.front());
            m_frameQueue.pop();
        }

        std::stringstream ss;
        ss << m_outputPath << "/frame_" << std::setw(6) << std::setfill('0') << m_frameCount++ << ".png";
        std::string filename = ss.str();

        try {
            // Save frameData as PNG using FileIO
            FileIO::flipVerticallyOnWrite(true);
            FileIO::saveAsPNG(filename, m_captureTarget->width, m_captureTarget->height, 4, frameData.data());
            std::cout << "Saved frame: " << filename << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error saving frame: " << filename << " - " << e.what() << std::endl;
        }
    }
}
