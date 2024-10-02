#include "BC4DepthStreamer.h"
#include <GL/glew.h>

BC4DepthStreamer::BC4DepthStreamer(const RenderTargetCreateParams &params, std::string receiverURL)
    : RenderTarget(params)
    , receiverURL(receiverURL)
    , streamer(receiverURL) {
    
    compressedSize = (params.width / 8) * (params.height / 8) * sizeof(Block);
    compressedData = std::vector<uint8_t>(sizeof(pose_id_t) + compressedSize);

    glGenBuffers(1, &bc4Buffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, bc4Buffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, compressedSize, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

#if !defined(__APPLE__) && !defined(__ANDROID__)
    cudautils::checkCudaDevice();
    CHECK_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&cudaResource, bc4Buffer, cudaGraphicsRegisterFlagsNone));

    running = true;
    dataSendingThread = std::thread(&BC4DepthStreamer::sendData, this);
#endif
}

BC4DepthStreamer::~BC4DepthStreamer() {
    close();
}

void BC4DepthStreamer::close() {
#if !defined(__APPLE__) && !defined(__ANDROID__)
    running = false;
    dataReady = true;
    cv.notify_one();

    if (dataSendingThread.joinable()) {
        dataSendingThread.join();
    }

    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(cudaResource));
#endif

    glDeleteBuffers(1, &bc4Buffer);
}

void BC4DepthStreamer::sendFrame(pose_id_t poseID) {
#if !defined(__APPLE__) && !defined(__ANDROID__)
    cudaArray* cudaBuffer;
    CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &cudaResource));
    CHECK_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&cudaBuffer, cudaResource, 0, 0));
    CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &cudaResource));

    {
        std::lock_guard<std::mutex> lock(m);
        cudaBufferQueue.push({poseID, cudaBuffer});
        dataReady = true;
    }
    cv.notify_one();
#else
    this->poseID = poseID;

    std::memcpy(compressedData.data(), &poseID, sizeof(pose_id_t));

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, bc4Buffer); // copies the BC4 compressed data from the GPU buffer to CPU memory.
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, compressedSize, compressedData.data() + sizeof(pose_id_t)); // 
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    float startTime = timeutils::getTimeMicros();

    streamer.send(compressedData);

    stats.timeToSendMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
    stats.bitrateMbps = ((compressedSize * 8) / timeutils::millisToSeconds(stats.timeToSendMs)) / MBPS_TO_BPS;
#endif
}

#if !defined(__APPLE__) && !defined(__ANDROID__)
void BC4DepthStreamer::sendData() {
    float prevTime = timeutils::getTimeMicros();

    while (running) {
        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock, [this] { return dataReady; });

        if (!running) break;

        dataReady = false;
        CudaBuffer cudaBufferStruct = cudaBufferQueue.front();
        cudaArray* cudaBuffer = cudaBufferStruct.buffer;
        cudaBufferQueue.pop();

        lock.unlock();

        float startTime = timeutils::getTimeMicros();


        // Uses CUDA to efficiently copy the compressed depth data from GPU to CPU memory.
        std::memcpy(compressedData.data(), &cudaBufferStruct.poseID, sizeof(pose_id_t));

        CHECK_CUDA_ERROR(cudaMemcpy2DFromArray(compressedData.data() + sizeof(pose_id_t),
                                               width * sizeof(Block) / 8,
                                               cudaBuffer,
                                               0, 0, width * sizeof(Block) / 8, height / 8,
                                               cudaMemcpyDeviceToHost));

        stats.timeToCopyFrameMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        float elapsedTimeSec = timeutils::microsToSeconds(timeutils::getTimeMicros() - prevTime);
        if (elapsedTimeSec < (1.0f / targetFrameRate)) {
            std::this_thread::sleep_for(
                std::chrono::microseconds(
                    (int)timeutils::secondsToMicros(1.0f / targetFrameRate - elapsedTimeSec)
                )
            );
        }

        startTime = timeutils::getTimeMicros();
        
        stats.timeToSendMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        stats.bitrateMbps = ((compressedSize * 8) / timeutils::millisToSeconds(stats.timeToSendMs)) / MBPS_TO_BPS;

        streamer.send(compressedData);

        prevTime = timeutils::getTimeMicros();
    }
}
#endif