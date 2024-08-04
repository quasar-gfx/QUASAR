#include <DepthStreamer.h>

DepthStreamer::DepthStreamer(const RenderTargetCreateParams &params, std::string receiverURL)
        : receiverURL(receiverURL)
        , imageSize(sizeof(pose_id_t) + params.width * params.height * sizeof(GLushort))
        , streamer(receiverURL)
        , RenderTarget(params) {
    data = std::vector<uint8_t>(imageSize);

    renderTargetCopy = new RenderTarget({
        .width = width,
        .height = height,
        .internalFormat = colorBuffer.internalFormat,
        .format = colorBuffer.format,
        .type = colorBuffer.type,
        .wrapS = colorBuffer.wrapS,
        .wrapT = colorBuffer.wrapT,
        .minFilter = colorBuffer.minFilter,
        .magFilter = colorBuffer.magFilter,
        .multiSampled = colorBuffer.multiSampled
    });

#ifndef __APPLE__
    cudautils::checkCudaDevice();
    // register opengl texture with cuda
    CHECK_CUDA_ERROR(cudaGraphicsGLRegisterImage(&cudaResource,
                                                 renderTargetCopy->colorBuffer.ID, GL_TEXTURE_2D,
                                                 cudaGraphicsRegisterFlagsReadOnly));

    // start data sending thread
    running = true;
    dataSendingThread = std::thread(&DepthStreamer::sendData, this);
#endif
}

void DepthStreamer::close() {
#ifndef __APPLE__
    running = false;

    // send dummy to unblock thread
    dataReady = true;
    cv.notify_one();

    if (dataSendingThread.joinable()) {
        dataSendingThread.join();
    }

    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(cudaResource));
#endif
}

void DepthStreamer::sendFrame(pose_id_t poseID) {
    renderTargetCopy->bind();
    blitToRenderTarget(*renderTargetCopy);
    renderTargetCopy->unbind();

#ifndef __APPLE__
    // add cuda buffer
    cudaArray* cudaBuffer;
    CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &cudaResource));
    CHECK_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&cudaBuffer, cudaResource, 0, 0));
    CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &cudaResource));

    {
        // lock mutex
        std::lock_guard<std::mutex> lock(m);

        CudaBuffer cudaBufferStruct = { poseID, cudaBuffer };
        cudaBufferQueue.push(cudaBufferStruct);

        // tell thread to send data
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

#ifndef __APPLE__
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

        float startTime = timeutils::getTimeMicros();

        // copy depth buffer to data
        CudaBuffer cudaBufferStruct = cudaBufferQueue.front();
        cudaArray* cudaBuffer = cudaBufferStruct.buffer;
        pose_id_t poseIDToSend = cudaBufferStruct.poseID;

        cudaBufferQueue.pop();

        lock.unlock();

        std::memcpy(data.data(), &poseIDToSend, sizeof(pose_id_t));

        CHECK_CUDA_ERROR(cudaMemcpy2DFromArray(data.data() + sizeof(pose_id_t), width * sizeof(GLushort),
                                               cudaBuffer,
                                               0, 0, width * sizeof(GLushort), height,
                                               cudaMemcpyDeviceToHost));

        stats.timeToCopyFrameMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        float elapsedTimeSec = timeutils::microsToSeconds(timeutils::getTimeMicros() - prevTime);
        if (elapsedTimeSec < (1.0f / targetFrameRate)) {
            std::this_thread::sleep_for(std::chrono::microseconds((int)timeutils::secondsToMicros(1.0f / targetFrameRate - elapsedTimeSec)));
        }
        stats.timeToSendMs = timeutils::microsToMillis(timeutils::getTimeMicros() - prevTime);

        stats.bitrateMbps = ((imageSize * 8) / timeutils::millisToSeconds(stats.timeToSendMs)) / MBPS_TO_BPS;

        streamer.send(data);

        prevTime = timeutils::getTimeMicros();
    }
}
#endif
