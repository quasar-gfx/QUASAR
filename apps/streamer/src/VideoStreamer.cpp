#include <VideoStreamer.h>

#undef av_err2str
#define av_err2str(errnum) av_make_error_string((char*)__builtin_alloca(AV_ERROR_MAX_STRING_SIZE), AV_ERROR_MAX_STRING_SIZE, errnum)

int VideoStreamer::start(Texture* texture, const std::string outputUrl) {
    this->sourceTexture = texture;
    this->outputUrl = outputUrl;

    /* BEGIN: Setup codec to encode output (video to URL) */
#ifdef __APPLE__
    const AVCodec *outputCodec = avcodec_find_encoder_by_name("h264_videotoolbox");
    std::cout << "Encoder: h264_videotoolbox" << std::endl;
#else
    const AVCodec *outputCodec = avcodec_find_encoder_by_name("h264_nvenc");
    std::cout << "Encoder: h264_nvenc" << std::endl;
#endif
    if (!outputCodec) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't allocate encoder.\n");
        return -1;
    }

    outputCodecContext = avcodec_alloc_context3(outputCodec);
    if (!outputCodecContext) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't allocate codec context.\n");
        return -1;
    }

    outputCodecContext->width = texture->width;
    outputCodecContext->height = texture->height;
    outputCodecContext->time_base = {1, frameRate};
    outputCodecContext->framerate = {frameRate, 1};
    outputCodecContext->pix_fmt = AV_PIX_FMT_YUV420P;
    outputCodecContext->bit_rate = 400000;

    // Set zero latency
    outputCodecContext->max_b_frames = 0;
    outputCodecContext->gop_size = 0;
    av_opt_set_int(outputCodecContext->priv_data, "zerolatency", 1, 0);
    av_opt_set_int(outputCodecContext->priv_data, "delay", 0, 0);

    int ret = avcodec_open2(outputCodecContext, outputCodec, nullptr);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't open codec: %s\n", av_err2str(ret));
        return ret;
    }
    /* END: Setup codec to encode output (video to URL) */

    /* BEGIN: Setup output (to write video to URL) */
    // Open output URL
    ret = avformat_alloc_output_context2(&outputFormatContext, nullptr, "mpegts", outputUrl.c_str());
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
    ret = avio_open(&outputFormatContext->pb, outputUrl.c_str(), AVIO_FLAG_WRITE);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Cannot open output URL\n");
        return ret;
    }

    ret = avformat_write_header(outputFormatContext, nullptr);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error writing header\n");
        return ret;
    }
    /* END: Setup output (to write video to URL) */

    conversionContext = sws_getContext(texture->width, texture->height, AV_PIX_FMT_RGBA,
                                                    texture->width, texture->height, AV_PIX_FMT_YUV420P,
                                                    SWS_BICUBIC, nullptr, nullptr, nullptr);
    if (!conversionContext) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Could not allocate conversion context\n");
        return -1;
    }

    rgbaData = new uint8_t[texture->width * texture->height * 4];

    return 0;
}

void VideoStreamer::sendFrame() {
    static uint64_t lastTime = av_gettime();

    // get frame from decoder
    AVFrame *frame = av_frame_alloc();
    frame->format = AV_PIX_FMT_YUV420P;
    frame->width = sourceTexture->width;
    frame->height = sourceTexture->height;

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

    sourceTexture->bind(0);
    glReadPixels(0, 0, sourceTexture->width, sourceTexture->height, GL_RGBA, GL_UNSIGNED_BYTE, rgbaData);
    sourceTexture->unbind();

    const uint8_t* srcData[] = { rgbaData };
    int srcStride[] = { static_cast<int>(sourceTexture->width * 4) }; // RGBA has 4 bytes per pixel

    sws_scale(conversionContext, srcData, srcStride, 0, sourceTexture->height, frame->data, frame->linesize);

    // send packet to encoder
    ret = avcodec_send_frame(outputCodecContext, frame);
    if (ret < 0 || ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Could not send frame to output encoder: %s\n", av_err2str(ret));
        return;
    }

    // get packet from encoder
    ret = avcodec_receive_packet(outputCodecContext, &outputPacket);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
        av_frame_free(&frame);
        av_packet_unref(&outputPacket);
        return;
    }
    else if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Could not receive frame from encoder: %s\n", av_err2str(ret));
        return;
    }

    outputPacket.pts = framesSent * (outputFormatContext->streams[0]->time_base.den) / frameRate;
    outputPacket.dts = outputPacket.pts;

    av_usleep(1.0f / frameRate * 1000000.0f - (av_gettime() - lastTime));

    // send packet to output URL
    ret = av_interleaved_write_frame(outputFormatContext, &outputPacket);
    av_packet_unref(&outputPacket);
    av_frame_free(&frame);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error writing frame\n");
        return;
    }

    framesSent++;

    lastTime = av_gettime();
}

void VideoStreamer::cleanup() {
    avio_closep(&outputFormatContext->pb);
    avformat_close_input(&outputFormatContext);
    avformat_free_context(outputFormatContext);
    sws_freeContext(conversionContext);
}
