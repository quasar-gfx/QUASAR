#include <iostream>

extern "C" {
#include <libavformat/avformat.h>
}

#undef av_err2str
#define av_err2str(errnum) av_make_error_string((char*)__builtin_alloca(AV_ERROR_MAX_STRING_SIZE), AV_ERROR_MAX_STRING_SIZE, errnum)

int main(int argc, char **argv) {
    AVFormatContext *inputContext = nullptr;

    AVPacket packet;
    int videoStreamIndex = -1;

    std::string inputUrl = "udp://localhost:1234"; // Specify the UDP input URL
    std::string outputFileName = "output.mp4";

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-i") && i + 1 < argc) {
            inputUrl = argv[i + 1];
            i++;
        }
        else if (!strcmp(argv[i], "-o") && i + 1 < argc) {
            outputFileName = argv[i + 1];
            i++;
        }
    }

    int ret = avformat_open_input(&inputContext, inputUrl.c_str(), nullptr, nullptr);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Cannot open input URL: %s\n", av_err2str(ret));
        return ret;
    }

    ret = avformat_find_stream_info(inputContext, nullptr);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Cannot find stream information: %s\n", av_err2str(ret));
        return ret;
    }

    // Find the video stream index
    for (int i = 0; i < inputContext->nb_streams; i++) {
        if (inputContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            videoStreamIndex = i;
            break;
        }
    }

    if (videoStreamIndex == -1) {
        av_log(nullptr, AV_LOG_ERROR, "No video stream found in the input URL\n");
        return AVERROR_STREAM_NOT_FOUND;
    }

    const AVOutputFormat *outputFormat = av_guess_format("mpegts", nullptr, nullptr);
    if (!outputFormat) {
        av_log(nullptr, AV_LOG_ERROR, "Cannot guess output format\n");
        return AVERROR_UNKNOWN;
    }

    AVFormatContext *outputContext = nullptr;
    ret = avformat_alloc_output_context2(&outputContext, outputFormat, nullptr, outputFileName.c_str());
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

    AVRational timeBase;
    AVRational streamTimeBase;

    // Get input and output stream time bases
    timeBase = inputContext->streams[videoStreamIndex]->time_base;
    streamTimeBase = outputContext->streams[videoStreamIndex]->time_base;

    int frame_index = 0;
    while (1) {
		AVStream *in_stream, *out_stream;

		ret = av_read_frame(inputContext, &packet);
		if (ret < 0) {
			break;
        }

		in_stream  = inputContext->streams[packet.stream_index];
		out_stream = outputContext->streams[packet.stream_index];

		// Convert PTS/DTS
		packet.pts = av_rescale_q_rnd(packet.pts, in_stream->time_base, out_stream->time_base, (AVRounding)(AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX));
		packet.dts = av_rescale_q_rnd(packet.dts, in_stream->time_base, out_stream->time_base, (AVRounding)(AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX));
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
            break;
        }
    }

    av_write_trailer(outputContext);
    avformat_close_input(&inputContext);
    avio_closep(&outputContext->pb);
    avformat_free_context(outputContext);

    return 0;
}

