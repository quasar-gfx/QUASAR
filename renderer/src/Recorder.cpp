#include <Recorder.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <filesystem> 
#include "Utils/FileIO.h"

Recorder::~Recorder() {
    stop();
}

void Recorder::start() {
    running = true;
    lastCaptureTime = std::chrono::steady_clock::now();
    for (int i = 0; i < NUM_SAVE_THREADS; ++i) {
        saveThreadPool.emplace_back(&Recorder::saveFrames, this);
    }
}

void Recorder::stop() {
    running = false;
    queueCV.notify_all();
    for (auto& thread : saveThreadPool) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    saveThreadPool.clear();
}

void Recorder::captureFrame(GeometryBuffer& gbuffer, Camera& camera) {
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
            frameQueue.push(FrameData{std::move(frameData), camera.getPosition(), camera.getRotationEuler()});
        }
        queueCV.notify_one();

        lastCaptureTime = currentTime;
    }
}

void Recorder::saveFrames() {
    if (!std::filesystem::exists(outputPath)) {
        std::filesystem::create_directories(outputPath);
    }
    
    while (running || !frameQueue.empty()) {
        FrameData frameData;
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            queueCV.wait(lock, [this] { return !frameQueue.empty() || !running; });
            if (!running && frameQueue.empty()) {
                break;
            }
            frameData = std::move(frameQueue.front());
            frameQueue.pop();
        }

        size_t currentFrame = frameCount++;
        std::stringstream ss;
        ss << outputPath << "/frame_" << std::setw(6) << std::setfill('0') << currentFrame << ".png";
        std::string filename = ss.str();

        try {
            FileIO::flipVerticallyOnWrite(true);
            FileIO::saveAsPNG(filename, captureTarget->width, captureTarget->height, 4, frameData.frame.data());
            
            {
                std::lock_guard<std::mutex> lock(queueMutex);
                std::ofstream pathFile(outputPath + "/camera_path.txt", std::ios::app);
                pathFile << std::fixed << std::setprecision(4)
                        << frameData.position.x << " "
                        << frameData.position.y << " "
                        << frameData.position.z << " "
                        << frameData.euler.x << " "
                        << frameData.euler.y << " "
                        << frameData.euler.z << std::endl;
                pathFile.close();
            }
            
            std::cout << "Saved frame: " << filename << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error saving frame: " << filename << " - " << e.what() << std::endl;
        }
    }
}

void Recorder::setOutputPath(const std::string& path) {
    outputPath = path;
}