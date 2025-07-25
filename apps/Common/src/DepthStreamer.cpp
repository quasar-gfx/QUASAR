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
        .internalFormat = colorBuffer.internalFormat,
        .format = colorBuffer.format,
        .type = colorBuffer.type,
        .wrapS = colorBuffer.wrapS,
        .wrapT = colorBuffer.wrapT,
        .minFilter = colorBuffer.minFilter,
        .magFilter = colorBuffer.magFilter,
        .multiSampled = colorBuffer.multiSampled,
    })
{
    spdlog::info("Created DepthStreamer that sends to URL: {}", receiverURL);

#if !defined(__APPLE__) && !defined(__ANDROID__)
    cudaImage.registerTexture(renderTargetCopy.colorBuffer);

    // Start data sending thread
    running = true;
    dataSendingThread = std::thread(&DepthStreamer::sendData, this);
#endif
}

DepthStreamer::~DepthStreamer() {
    close();
}

void DepthStreamer::close() {
#if !defined(__APPLE__) && !defined(__ANDROID__)
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
#if !defined(__APPLE__) && !defined(__ANDROID__)
    bind();
    blitToRenderTarget(renderTargetCopy);
    unbind();

    // Add cuda buffer
    cudaArray* cudaBuffer = cudaImage.getArray();
    {
        // Lock mutex
        std::lock_guard<std::mutex> lock(m);

        CudaBuffer cudaBufferStruct = { poseID, cudaBuffer };
        cudaBufferQueue.push(cudaBufferStruct);

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

#if !defined(__APPLE__) && !defined(__ANDROID__)
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
        cudaArray* cudaBuffer = cudaBufferStruct.buffer;
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
        stats.timeToCopyFrameMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startCopyTime);

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
