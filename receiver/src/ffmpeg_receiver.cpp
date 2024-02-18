#include "ffmpeg_receiver.hpp"

int FFmpegReceiver::init() {
    AVCodecParameters* codecParams = nullptr;

    // setup input (video from url) //
    std::cout << "Opening input URL" << std::endl;

    int ret = avformat_open_input(&inputContext, inputUrl.c_str(), nullptr, nullptr);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Cannot open input URL: %s\n", av_err2str(ret));
        return ret;
    }
    std::cout << "Input URL opened successfully" << std::endl;

    ret = avformat_find_stream_info(inputContext, nullptr);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Cannot find stream information: %s\n", av_err2str(ret));
        return ret;
    }

    // Find the video stream index
    for (int i = 0; i < inputContext->nb_streams; i++) {
        if (inputContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            videoStreamIndex = i;
            codecParams = inputContext->streams[i]->codecpar;
            break;
        }
    }

    if (videoStreamIndex == -1 || !codecParams) {
        av_log(nullptr, AV_LOG_ERROR, "No video stream found in the input URL.\n");
        return AVERROR_STREAM_NOT_FOUND;
    }

    // setup codec to decode input (video from URL) //

    const AVCodec *codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    if (!codec) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't allocate decoder.\n");
        return -1;
    }

    codecContext = avcodec_alloc_context3(codec);
    if (!codecContext) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't allocate codec context.\n");
        return -1;
    }

    ret = avcodec_parameters_to_context(codecContext, inputContext->streams[videoStreamIndex]->codecpar);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't copy codec parameters to context: %s\n", av_err2str(ret));
        return -1;
    }

    ret = avcodec_open2(codecContext, codec, nullptr);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't open codec: %s\n", av_err2str(ret));
        return -1;
    }

    // setup output (save to file) //

    // Open output file
    ret = avformat_alloc_output_context2(&outputContext, nullptr, "mpegts", outputFileName.c_str());
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Cannot allocate output context: %s\n", av_err2str(ret));
        return ret;
    }

    // Copy streams from input to output context
    for (unsigned int i = 0; i < inputContext->nb_streams; i++) {
        AVStream *stream = avformat_new_stream(outputContext, nullptr);
        if (!stream) {
            av_log(nullptr, AV_LOG_ERROR, "Failed allocating output stream\n");
            return AVERROR_UNKNOWN;
        }
        ret = avcodec_parameters_copy(stream->codecpar, inputContext->streams[i]->codecpar);
        if (ret < 0) {
            av_log(nullptr, AV_LOG_ERROR, "Failed to copy codec parameters\n");
            return ret;
        }
        stream->codecpar->codec_tag = 0;
    }

    ret = avio_open(&outputContext->pb, outputFileName.c_str(), AVIO_FLAG_WRITE);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Cannot open output file: %s\n", av_err2str(ret));
        return ret;
    }

    ret = avformat_write_header(outputContext, nullptr);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error writing header: %s\n", av_err2str(ret));
        return ret;
    }
    std::cout << "Output file opened successfully" << std::endl;

    return 0;
}

void FFmpegReceiver::receive() {
    // Get input and output stream time bases
    timeBase = inputContext->streams[videoStreamIndex]->time_base;
    streamTimeBase = outputContext->streams[videoStreamIndex]->time_base;

    int ret = avcodec_receive_packet(codecContext, &packet);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error reading frame: %s\n", av_err2str(ret));
        return;
    }

    in_stream  = inputContext->streams[packet.stream_index];
    out_stream = outputContext->streams[packet.stream_index];

    // Convert PTS/DTS
    packet.pts = av_rescale_q_rnd(packet.pts, in_stream->time_base, out_stream->time_base,
                                (AVRounding)(AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX));
    packet.dts = av_rescale_q_rnd(packet.dts, in_stream->time_base, out_stream->time_base,
                                (AVRounding)(AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX));
    packet.duration = av_rescale_q(packet.duration, in_stream->time_base, out_stream->time_base);
    packet.pos = -1;

    if (packet.stream_index == videoStreamIndex) {
        std::cout << "Received " << frame_index << " video frames from input URL" << std::endl;
        frame_index++;
    }

    ret = av_interleaved_write_frame(outputContext, &packet);
    av_packet_unref(&packet);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error writing frame: %s\n", av_err2str(ret));
        return;
    }
}
