#include <spdlog/spdlog.h>
#include <Streamers/DepthStreamer.h>

using namespace quasar;

DepthStreamer::DepthStreamer(const RenderTargetCreateParams& params, std::string receiverURL)
    : receiverURL(receiverURL)
    , imageSize(params.width * params.height * sizeof(GLushort))
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
    , DataStreamerTCP(receiverURL)
{
    spdlog::info("Created DepthStreamer that sends to URL: tcp://{}", receiverURL);

#if defined(QUASAR_HAS_CUDA)
    cudaGLImage.registerTexture(renderTargetCopy.colorTexture);

    if (!receiverURL.empty()) {
        running = true;
        dataSendingThread = std::thread(&DepthStreamer::sendData, this);
    }
#endif
}

DepthStreamer::~DepthStreamer() {
    stop();
}

void DepthStreamer::stop() {
#if defined(QUASAR_HAS_CUDA)
    running = false;

    if (dataSendingThread.joinable()) {
        dataSendingThread.join();
    }
#endif
}

void DepthStreamer::sendFrame(pose_id_t poseID) {
#if defined(QUASAR_HAS_CUDA)
    bind();
    blit(renderTargetCopy);
    unbind();

    cudaGLImage.map();
    cudaArray_t cudaBuffer = cudaGLImage.getArrayMapped();
    cudaGLImage.unmap();

    cudaBufferQueue.enqueue({ poseID, cudaBuffer });
#else
    this->poseID = poseID;

    std::memcpy(data.data(), &poseID, sizeof(pose_id_t));

    bind();
    glReadPixels(0, 0, width, height, GL_RED, GL_UNSIGNED_SHORT, data.data() + sizeof(pose_id_t));
    unbind();

    send(data);
#endif
}

#if defined(QUASAR_HAS_CUDA)
void DepthStreamer::sendData() {
    float prevTime = timeutils::getTimeMicros();

    while (running) {
        CudaBuffer cudaBufferStruct;
        if (!cudaBufferQueue.try_dequeue(cudaBufferStruct)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        cudaArray_t cudaBuffer = cudaBufferStruct.buffer;
        pose_id_t poseIDToSend = cudaBufferStruct.poseID;

        double startTransferTimeMs = timeutils::getTimeMicros();

        std::memcpy(data.data(), &poseIDToSend, sizeof(pose_id_t));

        CHECK_CUDA_ERROR(cudaMemcpy2DFromArray(data.data() + sizeof(pose_id_t),
                                               width * sizeof(GLushort),
                                               cudaBuffer,
                                               0, 0,
                                               width * sizeof(GLushort), height,
                                               cudaMemcpyDeviceToHost));

        stats.transferTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTransferTimeMs);

        double elapsedTimeSec = timeutils::microsToSeconds(timeutils::getTimeMicros() - prevTime);
        if (elapsedTimeSec < (1.0f / targetFrameRate)) {
            std::this_thread::sleep_for(std::chrono::microseconds(
                (int)timeutils::secondsToMicros(1.0f / targetFrameRate - elapsedTimeSec)));
        }

        stats.sendTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - prevTime);
        stats.bitrateMbps = ((data.size() + sizeof(pose_id_t)) * 8 / timeutils::millisToSeconds(stats.sendTimeMs)) / BYTES_PER_MEGABYTE;

        send(data);

        prevTime = timeutils::getTimeMicros();
    }
}
#endif
