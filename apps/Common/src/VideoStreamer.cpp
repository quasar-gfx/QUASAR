#include <TimeUtils.h>
#include <VideoStreamer.h>

#undef av_err2str
#define av_err2str(errnum) av_make_error_string((char*)__builtin_alloca(AV_ERROR_MAX_STRING_SIZE), AV_ERROR_MAX_STRING_SIZE, errnum)

static int interrupt_callback(void* ctx) {
    bool* shouldTerminatePtr = (bool*)ctx;
    bool shouldTerminate = (shouldTerminatePtr != nullptr) ? *shouldTerminatePtr : false;
    return shouldTerminate;
}

VideoStreamer::VideoStreamer(const RenderTargetCreateParams &params, const std::string &videoURL)
        : videoURL("udp://" + videoURL)
        , RenderTarget(params) {
    int ret;

#ifndef __APPLE__
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
    codecCtx->max_b_frames = 0;
    codecCtx->gop_size = 0;
    av_opt_set_int(codecCtx->priv_data, "zerolatency", 1, 0);
    av_opt_set_int(codecCtx->priv_data, "delay", 0, 0);

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

    videoStreamerThread = std::thread(&VideoStreamer::encodeAndSendFrames, this);
}

#ifndef __APPLE__
int VideoStreamer::initCuda() {
    CUdevice device = cudautils::findCudaDevice();
    // register opengl texture with cuda
    CHECK_CUDA_ERROR(cudaGraphicsGLRegisterImage(&cudaResource,
                                                 colorBuffer.ID, GL_TEXTURE_2D,
                                                 cudaGraphicsRegisterFlagsReadOnly));

    return 0;
}
#endif

void VideoStreamer::sendFrame(pose_id_t poseID) {
    /* Copy frame from OpenGL texture to AVFrame */
    int startCopyTime = timeutils::getCurrTimeMs();

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
#else
    {
        // lock mutex
        std::lock_guard<std::mutex> lock(m);

        bind();
        glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, rgbaData.data());
        unbind();

        this->poseID = poseID;
#endif

        // tell thread to send frame
        frameReady = true;
    }
    cv.notify_one();

    stats.timeToCopyFrameMs = (timeutils::getCurrTimeMs() - startCopyTime);
}

void VideoStreamer::encodeAndSendFrames() {
    sendFrames = true;

    int prevTime = timeutils::getCurrTimeMs();

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

#ifndef __APPLE__
        // copy opengl texture data to frame
        CudaBuffer cudaBufferStruct = cudaBufferQueue.front();
        cudaArray* cudaBuffer = cudaBufferStruct.buffer;
        pose_id_t poseIDToSend = cudaBufferStruct.poseID;

        cudaBufferQueue.pop();

        lock.unlock();

        CHECK_CUDA_ERROR(cudaMemcpy2DFromArray(rgbaData.data(), width * 4,
                                               cudaBuffer,
                                               0, 0, width * 4, height,
                                               cudaMemcpyDeviceToHost));

#else
        pose_id_t poseIDToSend = this->poseID;
        lock.unlock();
#endif

        // convert RGBA to YUV
        const uint8_t* srcData[] = { rgbaData.data() };
        int srcStride[] = { static_cast<int>(width * 4) }; // RGBA has 4 bytes per pixel

        sws_scale(swsCtx, srcData, srcStride, 0, height, frame->data, frame->linesize);

        /* Encode frame */
        {
            int startEncodeTime = timeutils::getCurrTimeMs();

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

            packet->pts = poseIDToSend; // framesSent * (outputFormatCtx->streams[0]->time_base.den) / targetFrameRate;
            packet->dts = packet->pts;

            stats.timeToEncodeMs = (timeutils::getCurrTimeMs() - startEncodeTime);
        }

        /* Send frame to output URL */
        {
            int startWriteTime = timeutils::getCurrTimeMs();

            // send packet to output URL
            ret = av_write_frame(outputFormatCtx, packet);
            if (ret < 0) {
                av_packet_unref(packet);
                av_log(nullptr, AV_LOG_ERROR, "Error writing frame\n");
                continue;
            }

            bytesSent = packet->size;

            av_packet_unref(packet);

            framesSent++;

            stats.timeToSendMs = (timeutils::getCurrTimeMs() - startWriteTime);
        }

        int elapsedTime = (timeutils::getCurrTimeMs() - prevTime);
        if (elapsedTime < (1.0f / targetFrameRate)) {
            av_usleep((1.0f / targetFrameRate) - elapsedTime);
        }
        stats.totalTimeToSendMs = (timeutils::getCurrTimeMs() - prevTime);

        stats.bitrateMbps = ((bytesSent * 8) / (stats.totalTimeToSendMs * MILLISECONDS_IN_SECOND));

        prevTime = timeutils::getCurrTimeMs();
    }
}

void VideoStreamer::cleanup() {
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

#ifndef __APPLE__
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(cudaResource));
#endif

    av_frame_free(&frame);
    av_packet_unref(packet);
    av_packet_free(&packet);
}
