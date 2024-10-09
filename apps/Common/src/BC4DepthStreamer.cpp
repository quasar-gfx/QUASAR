#include <BC4DepthStreamer.h>

#define THREADS_PER_LOCALGROUP 16
#define BLOCK_SIZE 8

BC4DepthStreamer::BC4DepthStreamer(const RenderTargetCreateParams &params, std::string receiverURL, std::string bc4CompressionShaderPath)
        : RenderTarget(params)
        , receiverURL(receiverURL)
        , streamer(receiverURL)
        , compressedSize((params.width / BLOCK_SIZE) * (params.height / BLOCK_SIZE))
        , bc4CompressionShader({
            .computeCodePath = bc4CompressionShaderPath,
            .defines = {
                "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
            }
        }) {
    data = std::vector<uint8_t>(sizeof(pose_id_t) + compressedSize * sizeof(Block));
    bc4CompressedBuffer = Buffer<Block>(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW, compressedSize, nullptr);

#if !defined(__APPLE__) && !defined(__ANDROID__)
    cudautils::checkCudaDevice();
    CHECK_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&cudaResource, bc4CompressedBuffer, cudaGraphicsRegisterFlagsNone));

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
}

void BC4DepthStreamer::compressBC4() {
    bc4CompressionShader.bind();
    bc4CompressionShader.setTexture(colorBuffer, 0);

    // round to nearest multiple of BLOCK_SIZE
    glm::uvec2 depthMapSize = glm::uvec2((width / BLOCK_SIZE) * BLOCK_SIZE, (height / BLOCK_SIZE) * BLOCK_SIZE);

    bc4CompressionShader.setVec2("depthMapSize", depthMapSize);
    bc4CompressionShader.setVec2("bc4DepthSize", glm::uvec2(depthMapSize.x / BLOCK_SIZE, depthMapSize.y / BLOCK_SIZE));
    bc4CompressionShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, bc4CompressedBuffer);

    bc4CompressionShader.dispatch(((depthMapSize.x / BLOCK_SIZE) + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                  ((depthMapSize.y / BLOCK_SIZE) + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
    bc4CompressionShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void BC4DepthStreamer::sendFrame(pose_id_t poseID) {
    compressBC4();

#if !defined(__APPLE__) && !defined(__ANDROID__)
    void* cudaPtr;
    size_t size;
    CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &cudaResource));
    CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(&cudaPtr, &size, cudaResource));
    CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &cudaResource));

    {
        std::lock_guard<std::mutex> lock(m);
        CudaBuffer newBuffer;
        newBuffer.poseID = poseID;
        newBuffer.buffer = cudaPtr;
        cudaBufferQueue.push(newBuffer);
        dataReady = true;
    }
    cv.notify_one();
#else
    this->poseID = poseID;

    std::memcpy(data.data(), &poseID, sizeof(pose_id_t));

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, bc4CompressedBuffer); // copies the BC4 compressed data from the GPU buffer to CPU memory.
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, compressedSize * sizeof(Block), data.data() + sizeof(pose_id_t)); //
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    streamer.send(data);
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

        // copy pose ID to the beginning of data
        std::memcpy(data.data(), &cudaBufferStruct.poseID, sizeof(pose_id_t));

        // copy compressed BC4 data from GPU to CPU
        CHECK_CUDA_ERROR(cudaMemcpy(data.data() + sizeof(pose_id_t),
                                    cudaPtr,
                                    compressedSize * sizeof(Block),
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
        stats.timeToSendMs = timeutils::microsToMillis(timeutils::getTimeMicros() - prevTime);

        stats.bitrateMbps = ((compressedSize * sizeof(Block)) / timeutils::millisToSeconds(stats.timeToSendMs)) / MBPS_TO_BPS;

        streamer.send(data);

        prevTime = timeutils::getTimeMicros();
    }
}
#endif
