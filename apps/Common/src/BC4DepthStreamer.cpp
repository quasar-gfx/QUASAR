#include <spdlog/spdlog.h>

#include <Utils/FileIO.h>
#include <BC4DepthStreamer.h>

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
{
    resize(width, height);

    bc4CompressedBuffer = Buffer({
        .target = GL_SHADER_STORAGE_BUFFER,
        .dataSize = sizeof(BC4Block),
        .numElems = compressedSize,
        .usage = GL_DYNAMIC_DRAW,
    });

#if defined(HAS_CUDA)
    cudaBufferBc4.registerBuffer(bc4CompressedBuffer);

    if (!receiverURL.empty()) {
        running = true;
        streamer = std::make_unique<DataStreamerTCP>(receiverURL);
        dataSendingThread = std::thread(&BC4DepthStreamer::sendData, this);
    }
#endif

    spdlog::info("Created BC4DepthStreamer that sends to URL: {}", receiverURL);
}

BC4DepthStreamer::~BC4DepthStreamer() {
    close();
}

void BC4DepthStreamer::close() {
#if defined(HAS_CUDA)
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
        copyToCPU();
        return applyCodec();
    }

    return 0;
}

void BC4DepthStreamer::copyToCPU(pose_id_t poseID, void* cudaPtr) {
    std::memcpy(data.data(), &poseID, sizeof(pose_id_t));

#if defined(HAS_CUDA)
    if (cudaPtr == nullptr) {
        CudaGLBuffer::registerHostBuffer(data.data(), sizeof(pose_id_t) + compressedSize * sizeof(BC4Block));
        cudaBufferBc4.copyToHostAsync(data.data() + sizeof(pose_id_t), compressedSize * sizeof(BC4Block));
        cudaBufferBc4.synchronize();
        CudaGLBuffer::unregisterHostBuffer(data.data());
    }
    else {
        CudaGLBuffer::registerHostBuffer(data.data(), sizeof(pose_id_t) + compressedSize * sizeof(BC4Block));
        cudaBufferBc4.synchronize();
        cudaStream_t stream = cudaBufferBc4.getStream();
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
    bc4CompressedBuffer.setData(compressedSize * sizeof(BC4Block), data.data() + sizeof(pose_id_t));
    bc4CompressedBuffer.unbind();
#endif
}

size_t BC4DepthStreamer::applyCodec() {
    double startTime = timeutils::getTimeMicros();

    size_t compressedSize = codec.compress(data.data(), compressedData, data.size());
    compressedData.resize(compressedSize);

    stats.timeToCompressMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
    stats.compressionRatio = static_cast<float>(data.size()) / compressedSize;

    return compressedSize;
}

void BC4DepthStreamer::saveToFile(const Path& filename) {
    double startTime = timeutils::getTimeMicros();
    FileIO::saveToBinaryFile(filename.str(), compressedData.data(), compressedData.size());

    spdlog::info("Saved {:.3f}MB in {:.3f}ms",
                 static_cast<double>(compressedData.size()) / BYTES_PER_MEGABYTE,
                 timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));
}

void BC4DepthStreamer::sendFrame(pose_id_t poseID) {
    compress();

#if defined(HAS_CUDA)
    void* cudaPtr = cudaBufferBc4.getPtr();
    cudaBufferQueue.enqueue({ poseID, cudaPtr });
#else
    double startTime = timeutils::getTimeMicros();

    size_t fullSize = sizeof(pose_id_t) + compressedSize * sizeof(BC4Block);
    data.resize(fullSize);

    std::memcpy(data.data(), &poseID, sizeof(pose_id_t));
    bc4CompressedBuffer.bind();
    bc4CompressedBuffer.getData(data.data() + sizeof(pose_id_t));
    bc4CompressedBuffer.unbind();

    stats.timeToTransferMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    applyCodec();

    streamer->send(compressedData);
#endif
}

#if defined(HAS_CUDA)
void BC4DepthStreamer::sendData() {
    time_t prevTime = timeutils::getTimeMicros();

    while (running) {
        CudaBuffer cudaBufferStruct;
        if (!cudaBufferQueue.try_dequeue(cudaBufferStruct)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        time_t startTransferTime = timeutils::getTimeMicros();

        std::memcpy(data.data(), &cudaBufferStruct.poseID, sizeof(pose_id_t));
        copyToCPU(cudaBufferStruct.poseID, cudaBufferStruct.buffer);

        stats.timeToTransferMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTransferTime);

        size_t compressedSize = applyCodec();

        double elapsedTimeSec = timeutils::microsToSeconds(timeutils::getTimeMicros() - prevTime);
        if (elapsedTimeSec < (1.0f / targetFrameRate)) {
            std::this_thread::sleep_for(
                std::chrono::microseconds(
                    (int)timeutils::secondsToMicros(1.0f / targetFrameRate - elapsedTimeSec)
                )
            );
        }

        streamer->send(compressedData);

        stats.timeToSendMs = timeutils::microsToMillis(timeutils::getTimeMicros() - prevTime);
        stats.bitrateMbps = ((compressedSize * 8.0) / timeutils::millisToSeconds(stats.timeToSendMs)) / BYTES_PER_MEGABYTE;

        prevTime = timeutils::getTimeMicros();
    }
}
#endif
