#include <spdlog/spdlog.h>

#include <Utils/FileIO.h>
#include <Streamers/BC4DepthStreamer.h>

#include <shaders_common.h>

#define THREADS_PER_LOCALGROUP 32

using namespace quasar;

BC4DepthStreamer::BC4DepthStreamer(const RenderTargetCreateParams& params, const std::string& receiverURL)
    : receiverURL(receiverURL)
    , width((params.width + BC4_BLOCK_SIZE - 1) / BC4_BLOCK_SIZE * BC4_BLOCK_SIZE)
    , height((params.height + BC4_BLOCK_SIZE - 1) / BC4_BLOCK_SIZE * BC4_BLOCK_SIZE)
    , compressedSize((width / BC4_BLOCK_SIZE) * (height / BC4_BLOCK_SIZE))
    , data(sizeof(pose_id_t) + compressedSize * sizeof(BC4Block))
    , bc4CompressionShader({
        .computeCodeData = SHADER_COMMON_BC4_COMPRESS_COMP,
        .computeCodeSize = SHADER_COMMON_BC4_COMPRESS_COMP_len,
        .defines = {
            "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
        }
    })
    , RenderTarget(params)
    , DataStreamerTCP(receiverURL)
{
    resize(width, height);

    bc4CompressedBuffer = Buffer({
        .target = GL_SHADER_STORAGE_BUFFER,
        .dataSize = sizeof(BC4Block),
        .numElems = compressedSize,
        .usage = GL_DYNAMIC_DRAW,
    });

#if defined(QUASAR_HAS_CUDA)
    cudaBufferBC4.registerBuffer(bc4CompressedBuffer);

    if (!receiverURL.empty()) {
        running = true;
        dataSendingThread = std::thread(&BC4DepthStreamer::sendData, this);
    }
#endif

    if (!receiverURL.empty()) {
        spdlog::info("Created BC4DepthStreamer that sends to URL: tcp://{}", receiverURL);
    }
}

BC4DepthStreamer::~BC4DepthStreamer() {
    stop();
}

void BC4DepthStreamer::stop() {
#if defined(QUASAR_HAS_CUDA)
    running = false;

    if (dataSendingThread.joinable()) {
        dataSendingThread.join();
    }
#endif
}

size_t BC4DepthStreamer::compress(bool compress) {
    bc4CompressionShader.bind();
    bc4CompressionShader.setTexture(colorTexture, 0);

    glm::uvec2 depthMapSize = glm::uvec2(width, height);

    bc4CompressionShader.setVec2("depthMapSize", depthMapSize);
    bc4CompressionShader.setVec2("bc4DepthSize", glm::uvec2(depthMapSize.x / BC4_BLOCK_SIZE, depthMapSize.y / BC4_BLOCK_SIZE));
    bc4CompressionShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, bc4CompressedBuffer);

    bc4CompressionShader.dispatch(((depthMapSize.x / BC4_BLOCK_SIZE) + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                  ((depthMapSize.y / BC4_BLOCK_SIZE) + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
    bc4CompressionShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    if (compress) {
        writeToMemory();
        return applyCodec();
    }

    return 0;
}

void BC4DepthStreamer::writeToMemory(pose_id_t poseID, void* cudaPtr) {
    std::memcpy(data.data(), &poseID, sizeof(pose_id_t));

#if defined(QUASAR_HAS_CUDA)
    if (cudaPtr == nullptr) {
        CudaGLBuffer::registerHostBuffer(data.data(), sizeof(pose_id_t) + compressedSize * sizeof(BC4Block));
        cudaBufferBC4.copyToHostAsync(data.data() + sizeof(pose_id_t), compressedSize * sizeof(BC4Block));
        cudaBufferBC4.synchronize();
        CudaGLBuffer::unregisterHostBuffer(data.data());
    }
    else {
        CudaGLBuffer::registerHostBuffer(data.data(), sizeof(pose_id_t) + compressedSize * sizeof(BC4Block));
        cudaBufferBC4.synchronize();
        cudaStream_t stream = cudaBufferBC4.getStream();
        CHECK_CUDA_ERROR(cudaMemcpyAsync(data.data() + sizeof(pose_id_t),
                                         cudaPtr,
                                         compressedSize * sizeof(BC4Block),
                                         cudaMemcpyDeviceToHost,
                                         stream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        CudaGLBuffer::unregisterHostBuffer(data.data());
    }
#else
    bc4CompressedBuffer.bind();
    bc4CompressedBuffer.getData(data.data() + sizeof(pose_id_t));
    bc4CompressedBuffer.unbind();
#endif
}

size_t BC4DepthStreamer::applyCodec() {
    double startTime = timeutils::getTimeMicros();

    size_t compressedSize = codec.compress(data.data(), compressedData, data.size());
    compressedData.resize(compressedSize);

    stats.compressTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
    stats.compressionRatio = static_cast<float>(data.size()) / compressedSize;

    return compressedSize;
}

void BC4DepthStreamer::writeToFile(const Path& filename) {
    double startTime = timeutils::getTimeMicros();
    FileIO::writeToBinaryFile(filename.str(), compressedData.data(), compressedData.size());

    spdlog::info("Saved {:.3f}MB in {:.3f}ms",
                 static_cast<double>(compressedData.size()) / BYTES_PER_MEGABYTE,
                 timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));
}

void BC4DepthStreamer::sendFrame(pose_id_t poseID) {
    compress();

#if defined(QUASAR_HAS_CUDA)
    void* cudaPtr = cudaBufferBC4.getPtr();
    cudaBufferQueue.enqueue({ poseID, cudaPtr });
#else
    double startTime = timeutils::getTimeMicros();

    size_t fullSize = sizeof(pose_id_t) + compressedSize * sizeof(BC4Block);
    data.resize(fullSize);

    std::memcpy(data.data(), &poseID, sizeof(pose_id_t));

    bc4CompressedBuffer.bind();
    bc4CompressedBuffer.getData(data.data() + sizeof(pose_id_t));
    bc4CompressedBuffer.unbind();

    stats.transferTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    applyCodec();

    send(compressedData);
#endif
}

#if defined(QUASAR_HAS_CUDA)
void BC4DepthStreamer::sendData() {
    time_t prevTime = timeutils::getTimeMicros();

    while (running) {
        CudaBuffer cudaBufferStruct;
        if (!cudaBufferQueue.try_dequeue(cudaBufferStruct)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        time_t startTransferTimeMs = timeutils::getTimeMicros();

        std::memcpy(data.data(), &cudaBufferStruct.poseID, sizeof(pose_id_t));
        writeToMemory(cudaBufferStruct.poseID, cudaBufferStruct.buffer);

        stats.transferTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTransferTimeMs);

        size_t compressedSize = applyCodec();

        double elapsedTimeSec = timeutils::microsToSeconds(timeutils::getTimeMicros() - prevTime);
        if (elapsedTimeSec < (1.0f / targetFrameRate)) {
            std::this_thread::sleep_for(
                std::chrono::microseconds(
                    (int)timeutils::secondsToMicros(1.0f / targetFrameRate - elapsedTimeSec)
                )
            );
        }

        send(compressedData);

        stats.sendTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - prevTime);
        stats.bitrateMbps = ((compressedSize * 8.0) / timeutils::millisToSeconds(stats.sendTimeMs)) / BYTES_PER_MEGABYTE;

        prevTime = timeutils::getTimeMicros();
    }
}
#endif
