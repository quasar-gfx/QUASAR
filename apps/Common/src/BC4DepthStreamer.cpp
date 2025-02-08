#include <spdlog/spdlog.h>

#include <BC4DepthStreamer.h>

#include <shaders_common.h>

#define THREADS_PER_LOCALGROUP 16
#define BLOCK_SIZE 8

BC4DepthStreamer::BC4DepthStreamer(const RenderTargetCreateParams &params, std::string receiverURL)
        : RenderTarget(params)
        , receiverURL(receiverURL)
        , streamer(receiverURL)
        , bc4CompressionShader({
            .computeCodeData = SHADER_COMMON_BC4COMPRESSION_COMP,
            .computeCodeSize = SHADER_COMMON_BC4COMPRESSION_COMP_len,
            .defines = {
                "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
            }
        }) {
    // round up to nearest multiple of BLOCK_SIZE
    width = (params.width + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    height = (params.height + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    resize(width, height);

    compressedSize = (width / BLOCK_SIZE) * (height / BLOCK_SIZE);
    data = std::vector<char>(sizeof(pose_id_t) + compressedSize * sizeof(Block));
    bc4CompressedBuffer = Buffer(GL_SHADER_STORAGE_BUFFER, compressedSize, sizeof(Block), nullptr, GL_DYNAMIC_DRAW);

#if !defined(__APPLE__) && !defined(__ANDROID__)
    cudautils::checkCudaDevice();
    CHECK_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&cudaResource, bc4CompressedBuffer, cudaGraphicsRegisterFlagsNone));

    running = true;
    dataSendingThread = std::thread(&BC4DepthStreamer::sendData, this);
#endif

    spdlog::info("Created BC4DepthStreamer that sends to URL: {}", receiverURL);
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

    glm::uvec2 depthMapSize = glm::uvec2(width, height);

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
    float startTime = timeutils::getTimeMicros();

    // Ensure data buffer has correct size
    size_t fullSize = sizeof(pose_id_t) + compressedSize * sizeof(Block);
    data.resize(fullSize);

    // Copy data
    std::memcpy(data.data(), &poseID, sizeof(pose_id_t));
    bc4CompressedBuffer.bind();
    bc4CompressedBuffer.getData(data.data() + sizeof(pose_id_t));
    bc4CompressedBuffer.unbind();

    stats.timeToCopyFrameMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    startTime = timeutils::getTimeMicros();

    // compress
    size_t maxSizeBytes = data.size();
    compressedData.resize(maxSizeBytes);
    compressor.compress(data.data(), compressedData, data.size());

    stats.timeToCompressMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
    stats.compressionRatio = static_cast<float>(data.size()) / compressedSize;

    streamer.send(compressedData);

    // spdlog::info("Frame Stats - Original: {} bytes, Compressed: {} bytes, Ratio: {}",
    //              data.size(), compressedSize, stats.compressionRatio);
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

        // Copy pose ID to the beginning of data
        std::memcpy(data.data(), &cudaBufferStruct.poseID, sizeof(pose_id_t));

        // Copy compressed BC4 data from GPU to CPU
        CHECK_CUDA_ERROR(cudaMemcpy(data.data() + sizeof(pose_id_t),
                                    cudaPtr,
                                    compressedSize * sizeof(Block),
                                    cudaMemcpyDeviceToHost));

        stats.timeToCopyFrameMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        startTime = timeutils::getTimeMicros();

        // compress
        size_t maxSizeBytes = data.size();
        compressedData.resize(maxSizeBytes);
        compressor.compress(data.data(), compressedData, data.size());

        stats.timeToCompressMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
        stats.compressionRatio = static_cast<float>(data.size()) / compressedData.size();

        float elapsedTimeSec = timeutils::microsToSeconds(timeutils::getTimeMicros() - prevTime);
        if (elapsedTimeSec < (1.0f / targetFrameRate)) {
            std::this_thread::sleep_for(
                std::chrono::microseconds(
                    (int)timeutils::secondsToMicros(1.0f / targetFrameRate - elapsedTimeSec)
                )
            );
        }

        // Send compressed data
        streamer.send(compressedData);

        stats.timeToSendMs = timeutils::microsToMillis(timeutils::getTimeMicros() - prevTime);
        stats.bitrateMbps = (compressedData.size() * 8 / timeutils::millisToSeconds(stats.timeToSendMs)) / BYTES_IN_MB;

        prevTime = timeutils::getTimeMicros();
    }
}
#endif
