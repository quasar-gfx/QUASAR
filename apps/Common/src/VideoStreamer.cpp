#include <VideoStreamer.h>

#undef av_err2str
#define av_err2str(errnum) av_make_error_string((char*)__builtin_alloca(AV_ERROR_MAX_STRING_SIZE), AV_ERROR_MAX_STRING_SIZE, errnum)

VideoStreamer::VideoStreamer(RenderTarget* renderTarget, const std::string &videoURL)
        : renderTarget(renderTarget)
        , width(renderTarget->width)
        , height(renderTarget->height)
        , videoURL("udp://" + videoURL) {
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

    codecCtx->width = width;
    codecCtx->height = height;
    codecCtx->time_base = {1, targetFrameRate};
    codecCtx->framerate = {targetFrameRate, 1};
    codecCtx->pix_fmt = videoPixelFormat;
    codecCtx->bit_rate = targetBitRate;

    // Set zero latency
    codecCtx->max_b_frames = 0;
    codecCtx->gop_size = 0;
    av_opt_set_int(codecCtx->priv_data, "zerolatency", 1, 0);
    av_opt_set_int(codecCtx->priv_data, "delay", 0, 0);

    int ret = avcodec_open2(codecCtx, outputCodec, nullptr);
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

    conversionCtx = sws_getContext(width, height, openglPixelFormat,
                                       width, height, videoPixelFormat,
                                       SWS_BICUBIC, nullptr, nullptr, nullptr);
    if (!conversionCtx) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Could not allocate conversion context\n");
        throw std::runtime_error("Video Streamer could not be created.");
    }

    rgbData = new uint8_t[width * height * 3];

    // initialize frame
    frame->format = videoPixelFormat;
    frame->width = width;
    frame->height = height;

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

    videoStreamerThread = std::thread(&VideoStreamer::encodeAndSendFrame, this);
}

void VideoStreamer::sendFrame(unsigned int poseID) {
    /* Copy frame from OpenGL texture to AVFrame (SLOW) */
    uint64_t startCopyTime = av_gettime();

    renderTarget->bind();
    glReadPixels(0, 0, renderTarget->width, renderTarget->height, GL_RGB, GL_UNSIGNED_BYTE, rgbData);
    renderTarget->unbind();

    const uint8_t* srcData[] = { rgbData };
    int srcStride[] = { static_cast<int>(renderTarget->width * 3) }; // RGB has 3 bytes per pixel

    // lock frame mutex
    std::lock_guard<std::mutex> lock(frameMutex);

    sws_scale(conversionCtx, srcData, srcStride, 0, renderTarget->height, frame->data, frame->linesize);
    this->poseID = poseID;

    // tell thread to send frame
    frameReady = true;
    cv.notify_one();

    stats.timeToCopyFrame = (av_gettime() - startCopyTime) / MICROSECONDS_IN_MILLISECOND;
}

void VideoStreamer::encodeAndSendFrame() {
    static uint64_t prevTime = av_gettime();

    int ret;
    while (true) {
        // wait for frame to be ready
        std::unique_lock<std::mutex> lock(frameMutex);
        cv.wait(lock, [&] { return frameReady; });

        frameReady = false;

        /* Encode frame */
        {
            uint64_t startEncodeTime = av_gettime();

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

            packet->pts = poseID; // framesSent * (outputFormatCtx->streams[0]->time_base.den) / targetFrameRate;
            packet->dts = packet->pts;

            stats.timeToEncode = (av_gettime() - startEncodeTime) / MICROSECONDS_IN_MILLISECOND;
        }

        /* Send frame to output URL */
        {
            uint64_t startWriteTime = av_gettime();

            // send packet to output URL
            ret = av_interleaved_write_frame(outputFormatCtx, packet);
            if (ret < 0) {
                av_log(nullptr, AV_LOG_ERROR, "Error writing frame\n");
                continue;
            }

            framesSent++;

            stats.timeToSendFrame = (av_gettime() - startWriteTime) / MICROSECONDS_IN_MILLISECOND;
        }

        uint64_t elapsedTime = (av_gettime() - prevTime);
        if (elapsedTime < (1.0f / targetFrameRate * MICROSECONDS_IN_MILLISECOND)) {
            av_usleep((1.0f / targetFrameRate * MICROSECONDS_IN_MILLISECOND) - elapsedTime);
        }
        stats.totalTimeToSendFrame = (av_gettime() - prevTime) / MICROSECONDS_IN_MILLISECOND;

        prevTime = av_gettime();
    }
}

void VideoStreamer::cleanup() {
    avio_closep(&outputFormatCtx->pb);
    avformat_close_input(&outputFormatCtx);
    avformat_free_context(outputFormatCtx);
    sws_freeContext(conversionCtx);
}
