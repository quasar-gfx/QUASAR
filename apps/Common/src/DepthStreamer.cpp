#include <spdlog/spdlog.h>

#include <DepthStreamer.h>

using namespace quasar;

DepthStreamer::DepthStreamer(const RenderTargetCreateParams& params, std::string receiverURL)
    : receiverURL(receiverURL)
    , imageSize(params.width * params.height * sizeof(GLushort))
    , streamer(receiverURL)
    , data(std::vector<char>(sizeof(pose_id_t) + imageSize))
    , RenderTarget(params)
    , renderTargetCopy({
        .width = width,
        .height = height,
        .internalFormat = colorTexture.internalFormat,
        .format = colorTexture.format,
        .type = colorTexture.type,
        .wrapS = colorTexture.wrapS,
        .wrapT = colorTexture.wrapT,
        .minFilter = colorTexture.minFilter,
        .magFilter = colorTexture.magFilter,
        .multiSampled = colorTexture.multiSampled,
    })
{
    spdlog::info("Created DepthStreamer that sends to URL: {}", receiverURL);

#if defined(HAS_CUDA)
    cudaGLImage.registerTexture(renderTargetCopy.colorTexture);

    // Start data sending thread
    running = true;
    dataSendingThread = std::thread(&DepthStreamer::sendData, this);
#endif
}

DepthStreamer::~DepthStreamer() {
    close();
}

void DepthStreamer::close() {
#if defined(HAS_CUDA)
    running = false;

    // Send dummy to unblock thread
    dataReady = true;
    cv.notify_one();

    if (dataSendingThread.joinable()) {
        dataSendingThread.join();
    }
#endif
}

void DepthStreamer::sendFrame(pose_id_t poseID) {
#if defined(HAS_CUDA)
    bind();
    blitToRenderTarget(renderTargetCopy);
    unbind();

    // Add cuda buffer to queue
    cudaGLImage.map();
    cudaArray_t cudaBuffer = cudaGLImage.getArrayMapped();
    cudaGLImage.unmap();
    {
        // Lock mutex
        std::lock_guard<std::mutex> lock(m);
        cudaBufferQueue.push({ poseID, cudaBuffer });

        // Tell thread to send data
        dataReady = true;
    }
    cv.notify_one();
#else
    this->poseID = poseID;

    std::memcpy(data.data(), &poseID, sizeof(pose_id_t));

    bind();
    glReadPixels(0, 0, width, height, GL_RED, GL_UNSIGNED_SHORT, data.data() + sizeof(pose_id_t));
    unbind();

    streamer.send(data);
#endif
}

#if defined(HAS_CUDA)
void DepthStreamer::sendData() {
    float prevTime = timeutils::getTimeMicros();

    while (true) {
        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock, [this] { return dataReady; });

        if (running) {
            dataReady = false;
        }
        else {
            break;
        }

        // Copy depth buffer to data
        CudaBuffer cudaBufferStruct = cudaBufferQueue.front();
        cudaArray_t cudaBuffer = cudaBufferStruct.buffer;
        pose_id_t poseIDToSend = cudaBufferStruct.poseID;

        cudaBufferQueue.pop();

        lock.unlock();

        time_t startCopyTime = timeutils::getTimeMicros();
        {
            std::memcpy(data.data(), &poseIDToSend, sizeof(pose_id_t));
            CHECK_CUDA_ERROR(cudaMemcpy2DFromArray(data.data() + sizeof(pose_id_t), width * sizeof(GLushort),
                                                   cudaBuffer,
                                                   0, 0, width * sizeof(GLushort), height,
                                                   cudaMemcpyDeviceToHost));
        }
        stats.timeToTransferMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startCopyTime);

        double elapsedTimeSec = timeutils::microsToSeconds(timeutils::getTimeMicros() - prevTime);
        if (elapsedTimeSec < (1.0f / targetFrameRate)) {
            std::this_thread::sleep_for(
                std::chrono::microseconds(
                    (int)timeutils::secondsToMicros(1.0f / targetFrameRate - elapsedTimeSec)
                )
            );
        }
        stats.timeToSendMs = timeutils::microsToMillis(timeutils::getTimeMicros() - prevTime);
        stats.bitrateMbps = ((data.size() + sizeof(pose_id_t)) * 8 / timeutils::millisToSeconds(stats.timeToSendMs)) / BYTES_PER_MEGABYTE;

        streamer.send(data);

        prevTime = timeutils::getTimeMicros();
    }
}
#endif
