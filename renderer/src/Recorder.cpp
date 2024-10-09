#include <Recorder.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include "Utils/FileIO.h"

Recorder::~Recorder() {
    stop();
}

void Recorder::start() {
    running = true;
    lastCaptureTime = std::chrono::steady_clock::now();
    saveThread = std::thread(&Recorder::saveFrames, this);
}

void Recorder::stop() {
    running = false;
    queueCV.notify_one();
    if (saveThread.joinable()) {
        saveThread.join();
    }
}

void Recorder::captureFrame(GeometryBuffer& gbuffer) {
    auto currentTime = std::chrono::steady_clock::now();
    if (currentTime - lastCaptureTime >= frameInterval) {
        std::vector<unsigned char> frameData(captureTarget->width * captureTarget->height * 4);
        
        // Bind the source RenderTarget
        gbuffer.colorBuffer.bind(0);
        // Read pixels directly from the source RenderTarget
        glReadPixels(0, 0, captureTarget->width, captureTarget->height, GL_RGBA, GL_UNSIGNED_BYTE, frameData.data());
        // Unbind the source RenderTarget
        gbuffer.colorBuffer.unbind();

        {
            std::lock_guard<std::mutex> lock(queueMutex);
            frameQueue.push(std::move(frameData));
        }
        queueCV.notify_one();

        lastCaptureTime = currentTime;
    }
}

void Recorder::saveFrames() {
    while (running || !frameQueue.empty()) {
        std::vector<unsigned char> frameData;
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            queueCV.wait(lock, [this] { return !frameQueue.empty() || !running; });
            if (!running && frameQueue.empty()) {
                break;
            }
            frameData = std::move(frameQueue.front());
            frameQueue.pop();
        }

        std::stringstream ss;
        ss << outputPath << "/frame_" << std::setw(6) << std::setfill('0') << frameCount++ << ".png";
        std::string filename = ss.str();

        try {
            // Save frameData as PNG using FileIO
            FileIO::flipVerticallyOnWrite(true);
            FileIO::saveAsPNG(filename, captureTarget->width, captureTarget->height, 4, frameData.data());
            std::cout << "Saved frame: " << filename << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error saving frame: " << filename << " - " << e.what() << std::endl;
        }
    }
}
