#include <iostream>
#include <chrono>
#include <thread>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/time.h>
#include <libavutil/opt.h>
}

// ffmpeg -re -i input.mp4 -c:v libx264 -preset ultrafast -tune zerolatency -f mpegts udp://localhost:1234

int main(int argc, char **argv) {
    AVFormatContext *inputContext = nullptr;
    AVFormatContext *outputContext = nullptr;

    AVCodecContext *codecContext = nullptr;

    AVPacket packet;
    int videoStreamIndex = -1;
    AVCodecParameters* codecParams = nullptr;

    std::string inputFileName = "input.mp4";
    std::string outputUrl = "udp://localhost:1234";

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-i") && i + 1 < argc) {
            inputFileName = argv[i + 1];
            i++;
        }
        else if (!strcmp(argv[i], "-o") && i + 1 < argc) {
            outputUrl = argv[i + 1];
            i++;
        }
    }

    // setup input (video from file) //

    int ret = avformat_open_input(&inputContext, inputFileName.c_str(), nullptr, nullptr);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Cannot open input file: %s\n", av_err2str(ret));
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
            codecParams = inputContext->streams[i]->codecpar;
            break;
        }
    }

    if (videoStreamIndex == -1 || !codecParams) {
        av_log(nullptr, AV_LOG_ERROR, "No video stream found in the input file.\n");
        return 1;
    }

    // setup codec to encode input (video from file) //

    const AVCodec *codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    if (!codec) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't allocate decoder.\n");
        return -1;
    }

    codecContext = avcodec_alloc_context3(codec);
    if (!codecContext) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't allocate codec context.\n");
        return -1;
    }

    codecContext->width = codecParams->width;
    codecContext->height = codecParams->height;
    codecContext->time_base = inputContext->streams[videoStreamIndex]->time_base;
    codecContext->framerate = inputContext->streams[videoStreamIndex]->avg_frame_rate;
    codecContext->pix_fmt = (AVPixelFormat)codecParams->format;
    codecContext->bit_rate = 400000;

    // Set zero latency
    codecContext->max_b_frames = 0;
    codecContext->gop_size = 0;
    av_opt_set_int(codecContext->priv_data, "zerolatency", 1, 0);
    av_opt_set_int(codecContext->priv_data, "delay", 0, 0);

    // ret = avcodec_parameters_to_context(codecContext, inputContext->streams[videoStreamIndex]->codecpar);
    // if (ret < 0) {
    //     av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't copy codec parameters to context: %s\n", av_err2str(ret));
    //     return -1;
    // }

    ret = avcodec_open2(codecContext, codec, nullptr);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't open codec: %s\n", av_err2str(ret));
        return -1;
    }

    // setup output (stream to URL) //

    // Open output URL
    ret = avformat_alloc_output_context2(&outputContext, nullptr, "mpegts", outputUrl.c_str());
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Could not allocate output context: %s\n", av_err2str(ret));
        return -1;
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

    AVStream *outputVideoStream = avformat_new_stream(outputContext, codec);
    if (!outputVideoStream) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Could not create new video stream.");
        return -1;
    }

    ret = avcodec_parameters_from_context(outputVideoStream->codecpar, codecContext);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Could not initialize output stream parameters: %s\n", av_err2str(ret));
        return -1;
    }

    // Open output URL
    ret = avio_open(&outputContext->pb, outputUrl.c_str(), AVIO_FLAG_WRITE);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Cannot open output URL\n");
        return ret;
    }

    ret = avformat_write_header(outputContext, nullptr);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error writing header\n");
        return ret;
    }

	int64_t start_time = av_gettime();

    int frame_index = 0;
    while (1) {
        ret = av_read_frame(inputContext, &packet);
        if (ret < 0) {
            av_log(nullptr, AV_LOG_ERROR, "Error reading frame\n");
            break;
        }

        if (packet.pts == AV_NOPTS_VALUE) {
            // Write PTS
            AVRational time_base1 = inputContext->streams[videoStreamIndex]->time_base;
            // Duration between 2 frames (us)
            int64_t calc_duration = (double)AV_TIME_BASE/av_q2d(inputContext->streams[videoStreamIndex]->r_frame_rate);
            // Parameters
            packet.pts = (double)(frame_index*calc_duration)/(double)(av_q2d(time_base1)*AV_TIME_BASE);
            packet.dts = packet.pts;
            packet.duration = (double)calc_duration/(double)(av_q2d(time_base1)*AV_TIME_BASE);
        }

		if (packet.stream_index == videoStreamIndex) {
            packet.stream_index = outputVideoStream->index;

			AVRational time_base = inputContext->streams[videoStreamIndex]->time_base;
			AVRational time_base_q = {1, AV_TIME_BASE};
			int64_t pts_time = av_rescale_q(packet.dts, time_base, time_base_q);
			int64_t now_time = av_gettime() - start_time;
			if (pts_time > now_time) {
				av_usleep(pts_time - now_time);
            }

            // packet.pts = av_rescale_q_rnd(packet.pts, in_stream->time_base, out_stream->time_base, (AVRounding)(AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX));
            // packet.dts = av_rescale_q_rnd(packet.dts, in_stream->time_base, out_stream->time_base, (AVRounding)(AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX));
            // packet.duration = av_rescale_q(packet.duration, in_stream->time_base, out_stream->time_base);
            // packet.pos = -1;

            std::cout << "Sending frame " << frame_index << " to " << outputUrl << std::endl;
			frame_index++;
		}

        ret = av_interleaved_write_frame(outputContext, &packet);
        av_packet_unref(&packet);
        if (ret < 0) {
            av_log(nullptr, AV_LOG_ERROR, "Error writing frame\n");
            break;
        }
    }

    std::cout << "Wrote: " << frame_index << std::endl;

    av_write_trailer(outputContext);
    avformat_close_input(&inputContext);
    avio_closep(&outputContext->pb);
    avformat_free_context(outputContext);

    return 0;
}
