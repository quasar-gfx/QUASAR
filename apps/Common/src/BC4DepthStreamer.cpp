#include "BC4DepthStreamer.h"
//#include <GL/glew.h>


BC4DepthStreamer::BC4DepthStreamer(const RenderTargetCreateParams &params, std::string receiverURL)
    : RenderTarget(params)
    , receiverURL(receiverURL)
    //,computerShader...(can have the compute shader init during the constructor stage)
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

void BC4DepthStreamer::compressBC4(const Texture& depthStencilBuffer, ComputeShader& bc4CompressShader, const glm::uvec2& windowSize) {
    bc4CompressShader.bind();
    bc4CompressShader.setTexture(depthStencilBuffer, 0);
    bc4CompressShader.setVec2("depthMapSize", windowSize);
    bc4CompressShader.setVec2("bc4DepthSize", glm::vec2(windowSize.x / 8, windowSize.y / 8));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bc4Buffer);
    bc4CompressShader.dispatch((windowSize.x / 8) / 16, (windowSize.y / 8) / 16, 1);
    bc4CompressShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

 // can have the compute shader passed as parameter or have one init in the constructor
void BC4DepthStreamer::sendFrame(pose_id_t poseID, const Texture& depthStencilBuffer, ComputeShader& bc4CompressShader, const glm::uvec2& windowSize) {
    compressBC4(depthStencilBuffer, bc4CompressShader, windowSize);
#if !defined(__APPLE__) && !defined(__ANDROID__) 
    // ------------------move the shader proces inside here from the mw_streamer, to make poseID match the buffer later--------
    void* cudaPtr;
    size_t size;
    CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &cudaResource));
    CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(&cudaPtr, &size, cudaResource));

    {
        std::lock_guard<std::mutex> lock(m);
        CudaBuffer newBuffer;
        newBuffer.poseID = poseID;
        newBuffer.buffer = cudaPtr;
        cudaBufferQueue.push(newBuffer);
        dataReady = true;
    }
    cv.notify_one();
    CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &cudaResource));
#else
    this->poseID = poseID;

    std::memcpy(compressedData.data(), &poseID, sizeof(pose_id_t));

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, bc4Buffer); // copies the BC4 compressed data from the GPU buffer to CPU memory.
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, compressedSize, compressedData.data() + sizeof(pose_id_t)); // 
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    float startTime = timeutils::getTimeMicros();

    streamer.send(compressedData);
    debugPrintData(compressedData); // Add this line to print the data

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
        void* cudaPtr = cudaBufferStruct.buffer;
        cudaBufferQueue.pop();

        lock.unlock();

        float startTime = timeutils::getTimeMicros();

        // Copy pose ID to the beginning of compressedData
        std::memcpy(compressedData.data(), &cudaBufferStruct.poseID, sizeof(pose_id_t));

        // Copy compressed BC4 data from GPU to CPU
        CHECK_CUDA_ERROR(cudaMemcpy(compressedData.data() + sizeof(pose_id_t),
                                    cudaPtr,
                                    compressedSize,
                                    cudaMemcpyDeviceToHost));
        
        debugPrintData(compressedData); // Add this line to print the data

        stats.timeToCopyFrameMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        startTime = timeutils::getTimeMicros();
        
        streamer.send(compressedData);

        stats.timeToSendMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
        stats.bitrateMbps = ((compressedData.size() * 8) / timeutils::millisToSeconds(stats.timeToSendMs)) / MBPS_TO_BPS;

        float elapsedTimeSec = timeutils::microsToSeconds(timeutils::getTimeMicros() - prevTime);
        if (elapsedTimeSec < (1.0f / targetFrameRate)) {
            std::this_thread::sleep_for(
                std::chrono::microseconds(
                    (int)timeutils::secondsToMicros(1.0f / targetFrameRate - elapsedTimeSec)
                )
            );
        }

        prevTime = timeutils::getTimeMicros();
    }
}
#endif