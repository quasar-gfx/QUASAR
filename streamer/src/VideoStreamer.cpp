#include <VideoStreamer.h>

#undef av_err2str
#define av_err2str(errnum) av_make_error_string((char*)__builtin_alloca(AV_ERROR_MAX_STRING_SIZE), AV_ERROR_MAX_STRING_SIZE, errnum)

int VideoStreamer::start(const std::string inputFileName, const std::string outputUrl) {
    this->inputFileName = inputFileName;
    this->outputUrl = outputUrl;

    /* BEGIN: Setup input (to read video from file) */
    int ret = avformat_open_input(&inputFormatContext, inputFileName.c_str(), nullptr, nullptr);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Cannot open input file: %s\n", av_err2str(ret));
        return ret;
    }

    ret = avformat_find_stream_info(inputFormatContext, nullptr);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Cannot find stream information: %s\n", av_err2str(ret));
        return ret;
    }

    // find the video stream index
    for (int i = 0; i < inputFormatContext->nb_streams; i++) {
        if (inputFormatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            videoStreamIndex = i;
            inputVideoStream = inputFormatContext->streams[i];
            break;
        }
    }

    if (videoStreamIndex == -1 || inputVideoStream == nullptr) {
        av_log(nullptr, AV_LOG_ERROR, "No video stream found in the input file.\n");
    }
    /* END: Setup input (to read video from file) */

    /* BEGIN: Setup codec to decode input (video from file) */
    const AVCodec *inputCodec = avcodec_find_decoder(inputVideoStream->codecpar->codec_id);
    if (!inputCodec) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't allocate decoder.\n");
        return -1;
    }

    inputCodecContext = avcodec_alloc_context3(inputCodec);
    if (!inputCodecContext) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't allocate codec context.\n");
        return -1;
    }

    ret = avcodec_parameters_to_context(inputCodecContext, inputVideoStream->codecpar);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't copy codec parameters to context: %s\n", av_err2str(ret));
        return ret;
    }

    ret = avcodec_open2(inputCodecContext, inputCodec, nullptr);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't open codec: %s\n", av_err2str(ret));
        return ret;
    }
    /* END: Setup codec to decode input (video from file) */

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

    outputCodecContext->width = inputVideoStream->codecpar->width;
    outputCodecContext->height = inputVideoStream->codecpar->height;
    outputCodecContext->time_base = inputFormatContext->streams[videoStreamIndex]->time_base;
    outputCodecContext->framerate = inputFormatContext->streams[videoStreamIndex]->avg_frame_rate;
    outputCodecContext->pix_fmt = (AVPixelFormat)inputVideoStream->codecpar->format;
    outputCodecContext->bit_rate = 400000;

    // Set zero latency
    // outputCodecContext->max_b_frames = 0;
    // outputCodecContext->gop_size = 0;
    // av_opt_set_int(outputCodecContext->priv_data, "zerolatency", 1, 0);
    // av_opt_set_int(outputCodecContext->priv_data, "delay", 0, 0);

    ret = avcodec_open2(outputCodecContext, outputCodec, nullptr);
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

    ret = avcodec_parameters_from_context(outputVideoStream->codecpar, inputCodecContext);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Could not initialize output stream parameters: %s\n", av_err2str(ret));
        return ret;
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

    videoStreamerThread = std::thread(&VideoStreamer::sendFrame, this);

    return 0;
}

void VideoStreamer::sendFrame() {
    static int64_t start_time = av_gettime();

    while (1) {
        // read frame from file
        int ret = av_read_frame(inputFormatContext, &inputPacket);
        if (ret < 0) {
            av_log(nullptr, AV_LOG_ERROR, "Error reading frame: %s\n", av_err2str(ret));
            return;
        }

        if (inputPacket.stream_index == videoStreamIndex) {
            // send packet to decoder
            ret = avcodec_send_packet(inputCodecContext, &inputPacket);
            if (ret < 0) {
                av_log(nullptr, AV_LOG_ERROR, "Error: Could not send packet to input decoder: %s\n", av_err2str(ret));
                return;
            }

            // get frame from decoder
            AVFrame *frame = av_frame_alloc();
            ret = avcodec_receive_frame(inputCodecContext, frame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                av_frame_free(&frame);
                av_packet_unref(&inputPacket);
                continue;
            }
            else if (ret < 0) {
                av_log(nullptr, AV_LOG_ERROR, "Error: Could not receive raw frame from input decoder: %s\n", av_err2str(ret));
                return;
            }

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
                continue;
            }
            else if (ret < 0) {
                av_log(nullptr, AV_LOG_ERROR, "Error: Could not receive frame from encoder: %s\n", av_err2str(ret));
                return;
            }

            outputPacket.stream_index = outputVideoStream->index;
            if (outputPacket.pts == AV_NOPTS_VALUE) {
                // write PTS
                AVRational time_base1 = inputVideoStream->time_base;
                // duration between 2 frames (us)
                int64_t calc_duration = (double)AV_TIME_BASE/av_q2d(inputVideoStream->r_frame_rate);
                // parameters
                outputPacket.pts = (double)(framesSent*calc_duration)/(double)(av_q2d(time_base1)*AV_TIME_BASE);
                outputPacket.dts = outputPacket.pts;
                outputPacket.duration = (double)calc_duration/(double)(av_q2d(time_base1)*AV_TIME_BASE);
            }

            // important to maintain FPS - delay
            AVRational time_base = inputVideoStream->time_base;
            AVRational time_base_q = {1, AV_TIME_BASE};
            int64_t pts_time = av_rescale_q(outputPacket.dts, time_base, time_base_q);
            int64_t now_time = av_gettime() - start_time;
            if (pts_time > now_time) {
                av_usleep(pts_time - now_time);
            }

            // convert PTS/DTS
            outputPacket.pts = av_rescale_q_rnd(inputPacket.pts, inputVideoStream->time_base, outputVideoStream->time_base,
                                            (AVRounding)(AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX));
            outputPacket.dts = av_rescale_q_rnd(inputPacket.dts, inputVideoStream->time_base, outputVideoStream->time_base,
                                            (AVRounding)(AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX));
            outputPacket.duration = av_rescale_q(inputPacket.duration, inputVideoStream->time_base, outputVideoStream->time_base);
            outputPacket.pos = -1;

            // send packet to output URL
            framesSent++;
            std::cout << "Sending frame " << framesSent << " to " << outputUrl << std::endl;
            ret = av_interleaved_write_frame(outputFormatContext, &outputPacket);
            av_packet_unref(&outputPacket);
            av_frame_free(&frame);
            if (ret < 0) {
                av_log(nullptr, AV_LOG_ERROR, "Error writing frame\n");
                return;
            }
        }
    }
}

void VideoStreamer::cleanup() {
    avformat_close_input(&inputFormatContext);
    avformat_free_context(inputFormatContext);
    avio_closep(&outputFormatContext->pb);
    avformat_close_input(&outputFormatContext);
    avformat_free_context(outputFormatContext);
}
