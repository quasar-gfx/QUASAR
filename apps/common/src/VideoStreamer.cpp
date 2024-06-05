#include <VideoStreamer.h>

#undef av_err2str
#define av_err2str(errnum) av_make_error_string((char*)__builtin_alloca(AV_ERROR_MAX_STRING_SIZE), AV_ERROR_MAX_STRING_SIZE, errnum)

int VideoStreamer::start(RenderTarget* renderTarget, const std::string videoURL) {
    this->renderTarget = renderTarget;
    this->videoURL = "udp://" + videoURL;

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
        return -1;
    }

    codecContext = avcodec_alloc_context3(outputCodec);
    if (!codecContext) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't allocate codec context.\n");
        return -1;
    }

    codecContext->width = renderTarget->width;
    codecContext->height = renderTarget->height;
    codecContext->time_base = {1, targetFrameRate};
    codecContext->framerate = {targetFrameRate, 1};
    codecContext->pix_fmt = this->pixelFormat;
    codecContext->bit_rate = 100000 * 1000;

    // Set zero latency
    codecContext->max_b_frames = 0;
    codecContext->gop_size = 0;
    av_opt_set_int(codecContext->priv_data, "zerolatency", 1, 0);
    av_opt_set_int(codecContext->priv_data, "delay", 0, 0);

    int ret = avcodec_open2(codecContext, outputCodec, nullptr);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't open codec: %s\n", av_err2str(ret));
        return ret;
    }

    /* Setup output (to write video to URL) */
    ret = avformat_alloc_output_context2(&outputFormatContext, nullptr, "mpegts", videoURL.c_str());
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Could not allocate output context: %s\n", av_err2str(ret));
        return ret;
    }

    outputVideoStream = avformat_new_stream(outputFormatContext, outputCodec);
    if (!outputVideoStream) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Could not create new video stream.\n");
        return -1;
    }

    // Open output URL
    ret = avio_open(&outputFormatContext->pb, this->videoURL.c_str(), AVIO_FLAG_WRITE);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Cannot open output URL\n");
        return ret;
    }

    ret = avformat_write_header(outputFormatContext, nullptr);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error writing header\n");
        return ret;
    }

    conversionContext = sws_getContext(renderTarget->width, renderTarget->height, AV_PIX_FMT_RGBA,
                                       renderTarget->width, renderTarget->height, this->pixelFormat,
                                       SWS_BICUBIC, nullptr, nullptr, nullptr);
    if (!conversionContext) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Could not allocate conversion context\n");
        return -1;
    }

    rgbaData = new uint8_t[renderTarget->width * renderTarget->height * 4];

    // initialize frame
    frame->format = this->pixelFormat;
    frame->width = renderTarget->width;
    frame->height = renderTarget->height;

    return 0;
}

void VideoStreamer::sendFrame(unsigned int poseId) {
    static uint64_t prevTime = av_gettime();

    int ret = av_frame_get_buffer(frame, 0);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Could not allocate frame data: %s\n", av_err2str(ret));
        return;
    }

    ret = av_frame_make_writable(frame);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Could not make frame writable: %s\n", av_err2str(ret));
        return;
    }

    /* Copy frame from OpenGL texture to AVFrame (SLOW) */
    {
        uint64_t startCopyTime = av_gettime();

        renderTarget->bind();
        glReadPixels(0, 0, renderTarget->width, renderTarget->height, GL_RGBA, GL_UNSIGNED_BYTE, rgbaData);
        renderTarget->unbind();

        const uint8_t* srcData[] = { rgbaData };
        int srcStride[] = { static_cast<int>(renderTarget->width * 4) }; // RGBA has 4 bytes per pixel

        sws_scale(conversionContext, srcData, srcStride, 0, renderTarget->height, frame->data, frame->linesize);

        stats.timeToCopyFrame = (av_gettime() - startCopyTime) / MICROSECONDS_IN_MILLISECOND;
    }

    /* Encode frame */
    {
        uint64_t startEncodeTime = av_gettime();

        // send packet to encoder
        ret = avcodec_send_frame(codecContext, frame);
        if (ret < 0 || ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            av_log(nullptr, AV_LOG_ERROR, "Error: Could not send frame to output encoder: %s\n", av_err2str(ret));
            return;
        }

        // get packet from encoder
        ret = avcodec_receive_packet(codecContext, packet);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            return;
        }
        else if (ret < 0) {
            av_log(nullptr, AV_LOG_ERROR, "Error: Could not receive frame from encoder: %s\n", av_err2str(ret));
            return;
        }

        packet->pts = poseId; // framesSent * (outputFormatContext->streams[0]->time_base.den) / targetFrameRate;
        packet->dts = packet->pts;

        stats.timeToEncode = (av_gettime() - startEncodeTime) / MICROSECONDS_IN_MILLISECOND;
    }

    /* Send frame to output URL */
    {
        uint64_t startWriteTime = av_gettime();

        // send packet to output URL
        ret = av_interleaved_write_frame(outputFormatContext, packet);
        if (ret < 0) {
            av_log(nullptr, AV_LOG_ERROR, "Error writing frame\n");
            return;
        }

        framesSent++;

        stats.timeToSendFrame = (av_gettime() - startWriteTime) / MICROSECONDS_IN_MILLISECOND;
    }

    uint64_t elapsedTime = (av_gettime() - prevTime);
    if (elapsedTime < (1.0f / targetFrameRate * MICROSECONDS_IN_MILLISECOND)) {
        av_usleep((1.0f / targetFrameRate * MICROSECONDS_IN_MILLISECOND) - elapsedTime);
    }
    stats.totalTimeToSendFrame = elapsedTime / MICROSECONDS_IN_MILLISECOND;

    prevTime = av_gettime();
}

void VideoStreamer::cleanup() {
    avio_closep(&outputFormatContext->pb);
    avformat_close_input(&outputFormatContext);
    avformat_free_context(outputFormatContext);
    sws_freeContext(conversionContext);
}
