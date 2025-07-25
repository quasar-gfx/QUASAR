#include <spdlog/spdlog.h>

#include <BC4DepthStreamer.h>

#include <shaders_common.h>

#include <Utils/FileIO.h>

#define THREADS_PER_LOCALGROUP 16

using namespace quasar;

BC4DepthStreamer::BC4DepthStreamer(const RenderTargetCreateParams& params, const std::string& receiverURL)
    : RenderTarget(params)
    , receiverURL(receiverURL)
    , streamer(receiverURL)
    , bc4CompressionShader({
        .computeCodeData = SHADER_COMMON_BC4_COMPRESS_COMP,
        .computeCodeSize = SHADER_COMMON_BC4_COMPRESS_COMP_len,
        .defines = {
            "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
    }})
{
    // Round up to nearest multiple of BC4_BLOCK_SIZE
    width = (params.width + BC4_BLOCK_SIZE - 1) / BC4_BLOCK_SIZE * BC4_BLOCK_SIZE;
    height = (params.height + BC4_BLOCK_SIZE - 1) / BC4_BLOCK_SIZE * BC4_BLOCK_SIZE;
    resize(width, height);

    compressedSize = (width / BC4_BLOCK_SIZE) * (height / BC4_BLOCK_SIZE);
    data = std::vector<char>(sizeof(pose_id_t) + compressedSize * sizeof(BC4Block));
    bc4CompressedBuffer = Buffer(GL_SHADER_STORAGE_BUFFER, compressedSize, sizeof(BC4Block), nullptr, GL_DYNAMIC_DRAW);

#if !defined(__APPLE__) && !defined(__ANDROID__)
        cudaBufferBc4.registerBuffer(bc4CompressedBuffer);

    if (!receiverURL.empty()) {
        running = true;
        dataSendingThread = std::thread(&BC4DepthStreamer::sendData, this);
    }
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
#endif
}

uint BC4DepthStreamer::compress(bool compress) {
    bc4CompressionShader.bind();
    bc4CompressionShader.setTexture(colorBuffer, 0);

    glm::uvec2 depthMapSize = glm::uvec2(width, height);

    bc4CompressionShader.setVec2("depthMapSize", depthMapSize);
    bc4CompressionShader.setVec2("bc4DepthSize", glm::uvec2(depthMapSize.x / BC4_BLOCK_SIZE, depthMapSize.y / BC4_BLOCK_SIZE));
    bc4CompressionShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, bc4CompressedBuffer);

    bc4CompressionShader.dispatch(((depthMapSize.x / BC4_BLOCK_SIZE) + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                  ((depthMapSize.y / BC4_BLOCK_SIZE) + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
    bc4CompressionShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    if (compress) {
        copyFrameToCPU();
        return applyCodec();
    }
    return 0;
}

void BC4DepthStreamer::copyFrameToCPU(pose_id_t poseID, void* cudaPtr) {
    // Copy pose ID to the beginning of data
    std::memcpy(data.data(), &poseID, sizeof(pose_id_t));

    // Copy compressed BC4 data from GPU to CPU
#if !defined(__APPLE__) && !defined(__ANDROID__)
    if (cudaPtr == nullptr) {
        cudaPtr = cudaBufferBc4.getPtr();
    }
    CHECK_CUDA_ERROR(cudaMemcpy(data.data() + sizeof(pose_id_t),
                                cudaPtr,
                                compressedSize * sizeof(BC4Block),
                                cudaMemcpyDeviceToHost));
#else
    bc4CompressedBuffer.bind();
    bc4CompressedBuffer.setData(compressedSize * sizeof(BC4Block), data.data() + sizeof(pose_id_t));
    bc4CompressedBuffer.unbind();
#endif
}

uint BC4DepthStreamer::applyCodec() {
    size_t maxSizeBytes = data.size();
    compressedData.resize(maxSizeBytes);
    uint compressedSize = codec.compress(data.data(), compressedData, data.size());
    compressedData.resize(compressedSize);
    return compressedSize;
}

void BC4DepthStreamer::saveToFile(const std::string& filename) {
    std::ofstream outFile(filename, std::ios::binary);
    if (outFile.is_open()) {
        outFile.write(compressedData.data(), compressedData.size());
        outFile.close();
    } else {
        spdlog::error("Failed to write compressed depth data to file");
    }
}

void BC4DepthStreamer::sendFrame(pose_id_t poseID) {
    compress();

#if !defined(__APPLE__) && !defined(__ANDROID__)
    void* cudaPtr = cudaBufferBc4.getPtr();

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
    double startTime = timeutils::getTimeMicros();

    // Ensure data buffer has correct size
    size_t fullSize = sizeof(pose_id_t) + compressedSize * sizeof(BC4Block);
    data.resize(fullSize);

    // Copy data
    std::memcpy(data.data(), &poseID, sizeof(pose_id_t));
    bc4CompressedBuffer.bind();
    bc4CompressedBuffer.getData(data.data() + sizeof(pose_id_t));
    bc4CompressedBuffer.unbind();

    stats.timeToCopyFrameMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    startTime = timeutils::getTimeMicros();

    // Compress
    applyCodec();

    stats.timeToCompressMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
    stats.compressionRatio = static_cast<float>(data.size()) / compressedSize;

    streamer.send(compressedData);
#endif
}

#if !defined(__APPLE__) && !defined(__ANDROID__)
void BC4DepthStreamer::sendData() {
    time_t prevTime = timeutils::getTimeMicros();

    while (running) {
        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock, [this] { return dataReady; });

        if (!running) break;

        dataReady = false;
        CudaBuffer cudaBufferStruct = cudaBufferQueue.front();
        void* cudaPtr = cudaBufferStruct.buffer;
        cudaBufferQueue.pop();

        lock.unlock();

        time_t startCopyTime = timeutils::getTimeMicros();
        {
            // Copy pose ID to the beginning of data
            std::memcpy(data.data(), &cudaBufferStruct.poseID, sizeof(pose_id_t));
            // Copy compressed BC4 data from GPU to CPU
            copyFrameToCPU(cudaBufferStruct.poseID, cudaPtr);
        }
        stats.timeToCopyFrameMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startCopyTime);

        // Compress even further
        time_t startCompressTime = timeutils::getTimeMicros();
        uint compressedSize = applyCodec();
        stats.timeToCompressMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startCompressTime);
        stats.compressionRatio = static_cast<float>(compressedSize) / (width * height * sizeof(float));

        double elapsedTimeSec = timeutils::microsToSeconds(timeutils::getTimeMicros() - prevTime);
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
        stats.bitrateMbps = ((compressedSize * 8) / timeutils::millisToSeconds(stats.timeToSendMs)) / BYTES_PER_MEGABYTE;

        prevTime = timeutils::getTimeMicros();
    }
}
#endif

using namespace quasar;
