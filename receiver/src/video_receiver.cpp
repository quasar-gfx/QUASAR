#include "glad/glad.h"

#include "video_receiver.hpp"

#undef av_err2str
#define av_err2str(errnum) av_make_error_string((char*)__builtin_alloca(AV_ERROR_MAX_STRING_SIZE), AV_ERROR_MAX_STRING_SIZE, errnum)

int VideoReceiver::init(const std::string inputUrl, int textureWidth, int textureHeight) {
    this->inputUrl = inputUrl;
    this->textureWidth = textureWidth;
    this->textureHeight = textureHeight;

    int ret = initFFMpeg();
    if (ret < 0) {
        return ret;
    }

    ret = initOutputTexture();
    if (ret < 0) {
        return ret;
    }

    return 0;
}

int VideoReceiver::initFFMpeg() {
    AVStream* inputVideoStream = nullptr;

    std::cout << "Waiting to receive video..." << std::endl;

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
    // const AVCodec *inputCodec = avcodec_find_decoder(AV_CODEC_ID_H264);
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
    /* END: Setup codec to decode input (video from URL) */

    return 0;
}

int VideoReceiver::initOutputTexture() {
    frameRGB = av_frame_alloc();

    int numBytes = av_image_get_buffer_size(AV_PIX_FMT_RGB24, textureWidth, textureHeight, 1);
    buffer = (uint8_t*)av_malloc(numBytes * sizeof(uint8_t));

    av_image_fill_arrays(frameRGB->data, frameRGB->linesize, buffer, AV_PIX_FMT_RGB24, textureWidth, textureHeight, 1);

    swsContext = sws_getContext(inputCodecContext->width, inputCodecContext->height, inputCodecContext->pix_fmt,
                                textureWidth, textureHeight, AV_PIX_FMT_RGB24, SWS_BILINEAR, nullptr, nullptr, nullptr);

    // create framebuffer for video frames
    glGenTextures(1, &textureVideoBuffer);
    glBindTexture(GL_TEXTURE_2D, textureVideoBuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, textureWidth, textureHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureVideoBuffer, 0);

    return 0;
}

int VideoReceiver::receive() {
    // read frame from URL
    int ret = av_read_frame(inputFormatContext, &packet);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error reading frame: %s\n", av_err2str(ret));
        return ret;
    }

    if (packet.stream_index == videoStreamIndex) {
        // send packet to decoder
        ret = avcodec_send_packet(inputCodecContext, &packet);
        if (ret < 0 || ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            av_log(nullptr, AV_LOG_ERROR, "Error: Could not send packet to input decoder: %s\n", av_err2str(ret));
            return ret;
        }

        // get frame from decoder
        AVFrame *frame = av_frame_alloc();
        ret = avcodec_receive_frame(inputCodecContext, frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            av_packet_unref(&packet);
            return 1;
        }
        else if (ret < 0) {
            av_log(nullptr, AV_LOG_ERROR, "Error: Could not receive raw frame from input decoder: %s\n", av_err2str(ret));
            return ret;
        }

        // resize video frame to fit output texture size
        sws_scale(swsContext, (uint8_t const* const*)frame->data, frame->linesize,
                    0, inputCodecContext->height, frameRGB->data, frameRGB->linesize);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, textureWidth, textureHeight, GL_RGB, GL_UNSIGNED_BYTE, frameRGB->data[0]);

        frameReceived++;
        std::cout << "Received " << frameReceived << " video frames from " << inputUrl << std::endl;
    }

    return 0;
}

void VideoReceiver::cleanup() {
    avformat_close_input(&inputFormatContext);
    avformat_free_context(inputFormatContext);
}

