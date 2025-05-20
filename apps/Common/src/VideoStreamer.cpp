#include <cstring>

#include <spdlog/spdlog.h>

#include <Utils/TimeUtils.h>
#include <VideoStreamer.h>

#undef av_err2str
#define av_err2str(errnum) av_make_error_string((char*)__builtin_alloca(AV_ERROR_MAX_STRING_SIZE), AV_ERROR_MAX_STRING_SIZE, errnum)

using namespace quasar;

static int interrupt_callback(void* ctx) {
    bool* shouldTerminatePtr = (bool*)ctx;
    bool shouldTerminate = (shouldTerminatePtr != nullptr) ? *shouldTerminatePtr : false;
    return shouldTerminate;
}

VideoStreamer::VideoStreamer(const RenderTargetCreateParams& params,
                             const std::string& videoURL,
                             int targetFrameRate,
                             int targetBitRateMbps,
                             const std::string& formatName)
        : targetFrameRate(targetFrameRate)
        , targetBitRate(targetBitRateMbps * BYTES_IN_MB)
        , formatName(formatName)
        , RenderTarget(params)
#if !defined(__APPLE__) && !defined(__ANDROID__)
        , cudaGLImage(colorBuffer)
#endif
        {
    this->videoURL = (formatName == "mpegts") ?
                        "udp://" + videoURL :
                            formatName + "://" + videoURL;
    this->videoWidth = width + poseIDOffset;
    this->videoHeight = height;

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


    /* Setup codec to encode output (video to URL) */
#ifdef __APPLE__
    std::string encoderName = "h264_videotoolbox";
#elif __linux__
    std::string encoderName = "h264_nvenc";
#else
    std::string encoderName = "libx264";
#endif
    auto outputCodec = avcodec_find_encoder_by_name(encoderName.c_str());
    spdlog::info("Encoder: {}", encoderName);
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
    codecCtx->width = videoWidth;
    codecCtx->height = videoHeight;
    codecCtx->time_base = {1, targetFrameRate};
    codecCtx->framerate = {targetFrameRate, 1};
    codecCtx->bit_rate = targetBitRate;
    codecCtx->gop_size = 60;    // One keyframe every second
    codecCtx->max_b_frames = 0; // No B-frames for low latency

    av_opt_set(codecCtx->priv_data, "preset", preset.c_str(), 0);
    av_opt_set(codecCtx->priv_data, "tune", tune.c_str(), 0);
    av_opt_set(codecCtx->priv_data, "zerolatency", "1", 0);
    av_opt_set_int(codecCtx->priv_data, "delay", 0, 0);

    ret = avcodec_open2(codecCtx, outputCodec, nullptr);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't open codec: %s\n", av_err2str(ret));
        throw std::runtime_error("Video Streamer could not be created.");
    }

    /* Setup output (to write video to URL) */
    ret = avformat_alloc_output_context2(&outputFormatCtx, nullptr, formatName.c_str(), videoURL.c_str());
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

    // Rgba to yuv conversion
    swsCtx = sws_getContext(videoWidth, videoHeight, rgbaPixelFormat,
                            videoWidth, videoHeight, videoPixelFormat,
                            SWS_BICUBIC, nullptr, nullptr, nullptr);
    if (!swsCtx) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Could not allocate conversion context\n");
        throw std::runtime_error("Video Streamer could not be created.");
    }

    rgbaVideoFrameData = std::vector<uint8_t>(videoWidth * videoHeight * 4);
#if defined(__APPLE__) || defined(__ANDROID__)
    openglFrameData = std::vector<uint8_t>(width * height * 4);
#endif

    /* Setup frame */
    frame->width = videoWidth;
    frame->height = videoHeight;
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

    /* Setup packet */
    ret = av_packet_make_writable(packet);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Could not make packet writable: %s\n", av_err2str(ret));
        return;
    }

    spdlog::info("Created VideoStreamer that sends to URL: {} ({})", videoURL, formatName);

    videoStreamerThread = std::thread(&VideoStreamer::encodeAndSendFrames, this);
}

void VideoStreamer::sendFrame(pose_id_t poseID) {
#if !defined(__APPLE__) && !defined(__ANDROID__)
    bind();
    blitToRenderTarget(*renderTargetCopy);
    unbind();

    // Add cuda buffer
    cudaArray* cudaBuffer = cudaGLImage.getArray();
    {
        // Lock mutex
        std::lock_guard<std::mutex> lock(m);

        CudaBuffer cudaBufferStruct = { poseID, cudaBuffer };
        cudaBufferQueue.push(cudaBufferStruct);
#else
    {
        // Lock mutex
        std::lock_guard<std::mutex> lock(m);

        this->poseID = poseID;

        bind();
        glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, openglFrameData.data());
        unbind();
#endif

        // Tell thread to send frame
        frameReady = true;
    }
    cv.notify_one();
}

void VideoStreamer::packPoseIDIntoVideoFrame(pose_id_t poseID) {
    for (int i = 0; i < poseIDOffset; i++) {
        uint8_t value = (poseID & (1 << i)) ? 255 : 0;
        for (int j = 0; j < videoHeight; j++) {
            int index = j * videoWidth * 4 + (videoWidth - 1 - i) * 4;
            rgbaVideoFrameData[index + 0] = value; // R
            rgbaVideoFrameData[index + 1] = value; // G
            rgbaVideoFrameData[index + 2] = value; // B
            rgbaVideoFrameData[index + 3] = value; // A
        }
    }
}

void VideoStreamer::encodeAndSendFrames() {
    sendFrames = true;

    time_t prevTime = timeutils::getTimeMicros();

    size_t bytesSent = 0;
    int ret;
    while (true) {
        // Wait for frame to be ready
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
        time_t startCopyTime = timeutils::getTimeMicros();

        // Copy opengl texture data to frame
        CudaBuffer cudaBufferStruct = cudaBufferQueue.front();
        cudaArray* cudaBuffer = cudaBufferStruct.buffer;
        pose_id_t poseIDToSend = cudaBufferStruct.poseID;

        cudaBufferQueue.pop();

        lock.unlock();

        CHECK_CUDA_ERROR(cudaMemcpy2DFromArray(rgbaVideoFrameData.data(),
                                               videoWidth * 4,
                                               cudaBuffer,
                                               0, 0,
                                               width * 4, height,
                                               cudaMemcpyDeviceToHost));

        stats.timeToCopyFrameMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startCopyTime);
#else
        pose_id_t poseIDToSend = this->poseID;

        for (int row = 0; row < height; row++) {
            int srcIndex = row * width * 4;
            int dstIndex = row * videoWidth * 4;
            std::memcpy(rgbaVideoFrameData.data() + dstIndex, openglFrameData.data() + srcIndex, width * 4);
        }

        lock.unlock();
#endif

        packPoseIDIntoVideoFrame(poseIDToSend);

        /* Convert RGBA to YUV */
        {
            const uint8_t* srcData[] = { rgbaVideoFrameData.data() };
            int srcStride[] = { static_cast<int>(videoWidth * 4) }; // RGBA has 4 bytes per pixel

            sws_scale(swsCtx, srcData, srcStride, 0, videoHeight, frame->data, frame->linesize);
        }

        /* Encode frame */
        {
            time_t startEncodeTime = timeutils::getTimeMicros();

            // Send frame to encoder
            ret = avcodec_send_frame(codecCtx, frame);
            if (ret < 0 || ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                av_log(nullptr, AV_LOG_ERROR, "Error: Could not send frame to output encoder: %s\n", av_err2str(ret));
                continue;
            }

            // Get packet from encoder
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
            time_t startWriteTime = timeutils::getTimeMicros();

            bytesSent = packet->size;

            // Send packet to output URL
            ret = av_interleaved_write_frame(outputFormatCtx, packet);
            av_packet_unref(packet);
            if (ret < 0) {
                av_log(nullptr, AV_LOG_ERROR, "Error writing frame\n");
                continue;
            }

            framesSent++;

            stats.timeToSendMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startWriteTime);
        }

        double elapsedTimeSec = timeutils::microsToSeconds(timeutils::getTimeMicros() - prevTime);
        if (elapsedTimeSec < (1.0f / targetFrameRate)) {
            std::this_thread::sleep_for(
                std::chrono::microseconds(
                    (int)timeutils::secondsToMicros(1.0f / targetFrameRate - elapsedTimeSec)
                )
            );
        }
        stats.totalTimeToSendMs = timeutils::microsToMillis(timeutils::getTimeMicros() - prevTime);
        stats.bitrateMbps = ((bytesSent * 8) / timeutils::millisToSeconds(stats.totalTimeToSendMs)) / BYTES_IN_MB;

        prevTime = timeutils::getTimeMicros();
    }
}

VideoStreamer::~VideoStreamer() {
    shouldTerminate = true;
    sendFrames = false;

    // Send dummy frame to unblock thread
    frameReady = true;
    cv.notify_one();

    if (videoStreamerThread.joinable()) {
        videoStreamerThread.join();
    }

    avio_closep(&outputFormatCtx->pb);
    avformat_close_input(&outputFormatCtx);
    avformat_free_context(outputFormatCtx);

    av_frame_free(&frame);
    av_packet_unref(packet);
    av_packet_free(&packet);
}
