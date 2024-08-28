#include <cstring>

#include <Utils/TimeUtils.h>
#include <VideoStreamer.h>

#undef av_err2str
#define av_err2str(errnum) av_make_error_string((char*)__builtin_alloca(AV_ERROR_MAX_STRING_SIZE), AV_ERROR_MAX_STRING_SIZE, errnum)

static int interrupt_callback(void* ctx) {
    bool* shouldTerminatePtr = (bool*)ctx;
    bool shouldTerminate = (shouldTerminatePtr != nullptr) ? *shouldTerminatePtr : false;
    return shouldTerminate;
}

VideoStreamer::VideoStreamer(const RenderTargetCreateParams &params, const std::string &videoURL, unsigned int targetBitRateMbps)
        : videoURL("udp://" + videoURL)
        , targetBitRate(targetBitRateMbps * MBPS_TO_BPS)
        , RenderTarget(params) {
    resize(params.width + sizeof(pose_id_t) * 8, params.height);

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

    int ret;

#if !defined(__APPLE__) && !defined(__ANDROID__)
    ret = initCuda();
    if (ret < 0) {
        throw std::runtime_error("Error: Couldn't initialize CUDA");
    }

#endif

    /* Setup codec to encode output (video to URL) */
#ifdef __APPLE__
    std::string encoderName = "h264_videotoolbox";
#elif __linux__
    std::string encoderName = "h264_nvenc";
#else
    std::string encoderName = "libx264";
#endif
    auto outputCodec = avcodec_find_encoder_by_name(encoderName.c_str());
    std::cout << "Encoder: " << encoderName << std::endl;
    if (!outputCodec) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't allocate encoder.\n");
        throw std::runtime_error("Video Streamer could not be created.");
    }

    codecCtx = avcodec_alloc_context3(outputCodec);
    if (!codecCtx) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't allocate codec context.\n");
        throw std::runtime_error("Video Streamer could not be created.");
    }

    codecCtx->pix_fmt = videoPixelFormat;
    codecCtx->width = width;
    codecCtx->height = height;
    codecCtx->time_base = {1, targetFrameRate};
    codecCtx->framerate = {targetFrameRate, 1};
    codecCtx->bit_rate = targetBitRate;

    // Set zero latency
    codecCtx->gop_size = 60;    // One keyframe every second
    codecCtx->max_b_frames = 0; // No B-frames for low latency
    av_opt_set_int(codecCtx->priv_data, "zerolatency", 1, 0); // Zero latency
    av_opt_set_int(codecCtx->priv_data, "delay", 0, 0);       // No delay

    ret = avcodec_open2(codecCtx, outputCodec, nullptr);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't open codec: %s\n", av_err2str(ret));
        throw std::runtime_error("Video Streamer could not be created.");
    }

    /* Setup output (to write video to URL) */
    ret = avformat_alloc_output_context2(&outputFormatCtx, nullptr, "mpegts", videoURL.c_str());
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Could not allocate output context: %s\n", av_err2str(ret));
        throw std::runtime_error("Video Streamer could not be created.");
    }

    outputFormatCtx->interrupt_callback.callback = interrupt_callback;
    outputFormatCtx->interrupt_callback.opaque = &shouldTerminate;

    outputVideoStream = avformat_new_stream(outputFormatCtx, outputCodec);
    if (!outputVideoStream) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Could not create new video stream.\n");
        throw std::runtime_error("Video Streamer could not be created.");
    }

    outputVideoStream->time_base = codecCtx->time_base;

    ret = avcodec_parameters_from_context(outputVideoStream->codecpar, codecCtx);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Could not initialize stream codec parameters: %s\n", av_err2str(ret));
        throw std::runtime_error("Video Streamer could not be created.");
    }

    // Open output URL
    ret = avio_open(&outputFormatCtx->pb, this->videoURL.c_str(), AVIO_FLAG_WRITE);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Cannot open output URL\n");
        throw std::runtime_error("Video Streamer could not be created.");
    }

    ret = avformat_write_header(outputFormatCtx, nullptr);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error writing header\n");
        throw std::runtime_error("Video Streamer could not be created.");
    }

    swsCtx = sws_getContext(width, height, bufferPixelFormat,
                            width, height, videoPixelFormat,
                            SWS_BICUBIC, nullptr, nullptr, nullptr);
    if (!swsCtx) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Could not allocate conversion context\n");
        throw std::runtime_error("Video Streamer could not be created.");
    }

    rgbaData = std::vector<uint8_t>(width * height * 4);

    /* setup frame */
    frame->width = width;
    frame->height = height;
    frame->format = videoPixelFormat;
    ret = av_frame_get_buffer(frame, 0);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Could not allocate frame data: %s\n", av_err2str(ret));
        return;
    }

    ret = av_frame_make_writable(frame);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Could not make frame writable: %s\n", av_err2str(ret));
        return;
    }

    /* setup packet */
    ret = av_packet_make_writable(packet);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Could not make packet writable: %s\n", av_err2str(ret));
        return;
    }

    videoStreamerThread = std::thread(&VideoStreamer::encodeAndSendFrames, this);
}

#if !defined(__APPLE__) && !defined(__ANDROID__)
int VideoStreamer::initCuda() {
    cudautils::checkCudaDevice();
    // register opengl texture with cuda
    CHECK_CUDA_ERROR(cudaGraphicsGLRegisterImage(&cudaResource,
                                                 renderTargetCopy->colorBuffer.ID, GL_TEXTURE_2D,
                                                 cudaGraphicsRegisterFlagsReadOnly));
    return 0;
}
#endif

void VideoStreamer::sendFrame(pose_id_t poseID) {
    bind();
    blitToRenderTarget(*renderTargetCopy);
    unbind();

#if !defined(__APPLE__) && !defined(__ANDROID__)
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
#else
    {
        // lock mutex
        std::lock_guard<std::mutex> lock(m);

        this->poseID = poseID;

        bind();
        glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, rgbaData.data());
        unbind();
#endif

        // tell thread to send frame
        frameReady = true;
    }
    cv.notify_one();
}

void VideoStreamer::encodeAndSendFrames() {
    sendFrames = true;

    int prevTime = timeutils::getTimeMicros();

    size_t bytesSent = 0;
    int ret;
    while (true) {
        // wait for frame to be ready
        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock, [this] { return frameReady; });

        if (sendFrames) {
            frameReady = false;
        }
        else {
            break;
        }

#if !defined(__APPLE__) && !defined(__ANDROID__)
        /* Copy frame from OpenGL texture to AVFrame */
        int startCopyTime = timeutils::getTimeMicros();

        // copy opengl texture data to frame
        CudaBuffer cudaBufferStruct = cudaBufferQueue.front();
        cudaArray* cudaBuffer = cudaBufferStruct.buffer;
        pose_id_t poseIDToSend = cudaBufferStruct.poseID;

        cudaBufferQueue.pop();

        lock.unlock();

        CHECK_CUDA_ERROR(cudaMemcpy2DFromArray(rgbaData.data(), width * 4,
                                               cudaBuffer,
                                               0, 0,
                                               width * 4, height,
                                               cudaMemcpyDeviceToHost));

        stats.timeToCopyFrameMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startCopyTime);
#else
        pose_id_t poseIDToSend = this->poseID;
        lock.unlock();
#endif

        for (int i = 0; i < sizeof(pose_id_t) * 8; i++) {
            uint8_t value = (poseIDToSend & (1 << i)) ? 255 : 0;
            for (int j = 0; j < height; j++) {
                int index = j * width * 4 + (width - 1 - i) * 4;
                rgbaData[index + 0] = value; // R
                rgbaData[index + 1] = value; // G
                rgbaData[index + 2] = value; // B
                rgbaData[index + 3] = value; // A
            }
        }

        // convert RGBA to YUV
        const uint8_t* srcData[] = { rgbaData.data() };
        int srcStride[] = { static_cast<int>(width * 4) }; // RGBA has 4 bytes per pixel

        sws_scale(swsCtx, srcData, srcStride, 0, height, frame->data, frame->linesize);

        /* Encode frame */
        {
            int startEncodeTime = timeutils::getTimeMicros();

            // send frame to encoder
            ret = avcodec_send_frame(codecCtx, frame);
            if (ret < 0 || ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                av_log(nullptr, AV_LOG_ERROR, "Error: Could not send frame to output encoder: %s\n", av_err2str(ret));
                continue;
            }

            // get packet from encoder
            ret = avcodec_receive_packet(codecCtx, packet);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                continue;
            }
            else if (ret < 0) {
                av_log(nullptr, AV_LOG_ERROR, "Error: Could not receive frame from encoder: %s\n", av_err2str(ret));
                continue;
            }

            AVRational timeBase = outputFormatCtx->streams[videoStreamIndex]->time_base;
            packet->pts = av_rescale_q(framesSent, (AVRational){1, targetFrameRate}, timeBase);
            packet->dts = packet->pts;

            stats.timeToEncodeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startEncodeTime);
        }

        /* Send frame to output URL */
        {
            int startWriteTime = timeutils::getTimeMicros();

            bytesSent = packet->size;

            // send packet to output URL
            ret = av_interleaved_write_frame(outputFormatCtx, packet);
            if (ret < 0) {
                av_packet_unref(packet);
                av_log(nullptr, AV_LOG_ERROR, "Error writing frame\n");
                continue;
            }

            av_packet_unref(packet);

            framesSent++;

            stats.timeToSendMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startWriteTime);
        }

        float elapsedTimeSec = timeutils::microsToSeconds(timeutils::getTimeMicros() - prevTime);
        if (elapsedTimeSec < (1.0f / targetFrameRate)) {
            std::this_thread::sleep_for(
                std::chrono::microseconds(
                    (int)timeutils::secondsToMicros(1.0f / targetFrameRate - elapsedTimeSec)
                )
            );
        }
        stats.totalTimeToSendMs = timeutils::microsToMillis(timeutils::getTimeMicros() - prevTime);

        stats.bitrateMbps = ((bytesSent * 8) / timeutils::millisToSeconds(stats.totalTimeToSendMs)) / MBPS_TO_BPS;

        prevTime = timeutils::getTimeMicros();
    }
}

VideoStreamer::~VideoStreamer() {
    shouldTerminate = true;
    sendFrames = false;

    // send dummy frame to unblock thread
    frameReady = true;
    cv.notify_one();

    if (videoStreamerThread.joinable()) {
        videoStreamerThread.join();
    }

    avio_closep(&outputFormatCtx->pb);
    avformat_close_input(&outputFormatCtx);
    avformat_free_context(outputFormatCtx);

#if !defined(__APPLE__) && !defined(__ANDROID__)
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(cudaResource));
#endif

    av_frame_free(&frame);
    av_packet_unref(packet);
    av_packet_free(&packet);
}
