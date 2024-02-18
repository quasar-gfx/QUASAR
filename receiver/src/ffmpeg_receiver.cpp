#include "ffmpeg_receiver.hpp"

#undef av_err2str
#define av_err2str(errnum) av_make_error_string((char*)__builtin_alloca(AV_ERROR_MAX_STRING_SIZE), AV_ERROR_MAX_STRING_SIZE, errnum)

int FFmpegReceiver::init() {
    AVStream *inputVideoStream = nullptr;

    /* BEGIN: Setup input (to read video from url) */
    int ret = avformat_open_input(&inputFormatContext, inputUrl.c_str(), nullptr, nullptr);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Cannot open input URL: %s\n", av_err2str(ret));
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
        av_log(nullptr, AV_LOG_ERROR, "No video stream found in the input URL.\n");
        return AVERROR_STREAM_NOT_FOUND;
    }
    /* END: Setup input (to read video from url) */

    /* BEGIN: Setup codec to decode input (video from URL) */
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
        return -1;
    }

    ret = avcodec_open2(inputCodecContext, inputCodec, nullptr);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't open codec: %s\n", av_err2str(ret));
        return -1;
    }
    /* END: Setup codec to decode input (video from URL) */

    /* BEGIN: Setup output (save to file) */
    const AVOutputFormat *outputFormat = av_guess_format("mpegts", nullptr, nullptr);
    if (!outputFormat) {
        av_log(nullptr, AV_LOG_ERROR, "Cannot guess output format\n");
        return AVERROR_UNKNOWN;
    }

    ret = avformat_alloc_output_context2(&outputFormatContext, outputFormat, "mpegts", outputFileName.c_str());
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Cannot allocate output context: %s\n", av_err2str(ret));
        return ret;
    }

    // Copy streams from input to output context
    for (unsigned int i = 0; i < inputFormatContext->nb_streams; i++) {
        AVStream *stream = avformat_new_stream(outputFormatContext, nullptr);
        if (!stream) {
            av_log(nullptr, AV_LOG_ERROR, "Failed allocating output stream\n");
            return AVERROR_UNKNOWN;
        }
        ret = avcodec_parameters_copy(stream->codecpar, inputFormatContext->streams[i]->codecpar);
        if (ret < 0) {
            av_log(nullptr, AV_LOG_ERROR, "Failed to copy codec parameters\n");
            return ret;
        }
        stream->codecpar->codec_tag = 0;
    }

    // Open output file
    ret = avio_open(&outputFormatContext->pb, outputFileName.c_str(), AVIO_FLAG_WRITE);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Cannot open output file: %s\n", av_err2str(ret));
        return ret;
    }

    ret = avformat_write_header(outputFormatContext, nullptr);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error writing header: %s\n", av_err2str(ret));
        return ret;
    }
    /* END: Setup output (save to file) */

    return 0;
}

void FFmpegReceiver::receive() {
    timeBase = inputFormatContext->streams[videoStreamIndex]->time_base;
    streamTimeBase = outputFormatContext->streams[videoStreamIndex]->time_base;

    // read frame from URL
    int ret = av_read_frame(inputFormatContext, &packet);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error reading frame: %s\n", av_err2str(ret));
        return;
    }

    if (packet.stream_index == videoStreamIndex) {
		inputStream  = inputFormatContext->streams[packet.stream_index];
		outputStream = outputFormatContext->streams[packet.stream_index];

        // Convert PTS/DTS
        packet.pts = av_rescale_q_rnd(packet.pts, inputStream->time_base, outputStream->time_base,
                                        (AVRounding)(AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX));
        packet.dts = av_rescale_q_rnd(packet.dts, inputStream->time_base, outputStream->time_base,
                                        (AVRounding)(AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX));
        packet.duration = av_rescale_q(packet.duration, inputStream->time_base, outputStream->time_base);
        packet.pos = -1;

        frame_index++;
        std::cout << "Received " << frame_index << " video frames from input URL" << std::endl;

        ret = av_interleaved_write_frame(outputFormatContext, &packet);
        av_packet_unref(&packet);
        if (ret < 0) {
            av_log(nullptr, AV_LOG_ERROR, "Error writing frame: %s\n", av_err2str(ret));
            return;
        }
    }
}

void FFmpegReceiver::cleanup() {
    av_write_trailer(outputFormatContext);
    avformat_close_input(&inputFormatContext);
    avio_closep(&outputFormatContext->pb);
    avformat_free_context(inputFormatContext);
    avformat_free_context(outputFormatContext);
}

