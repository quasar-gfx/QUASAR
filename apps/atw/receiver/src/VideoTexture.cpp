#include <glad/glad.h>

#include <VideoTexture.h>

#undef av_err2str
#define av_err2str(errnum) av_make_error_string((char*)__builtin_alloca(AV_ERROR_MAX_STRING_SIZE), AV_ERROR_MAX_STRING_SIZE, errnum)

void VideoTexture::initVideo(const std::string &inputUrl) {
    this->inputUrl = inputUrl + "?overrun_nonfatal=1&fifo_size=50000000";
    videoReceiverThread = std::thread(&VideoTexture::receiveVideo, this);
}

int VideoTexture::initFFMpeg() {
    AVStream* inputVideoStream = nullptr;

    std::cout << "Waiting to receive video..." << std::endl;

    /* BEGIN: Setup input (to read video from url) */
    int ret = avformat_open_input(&inputFormatContext, inputUrl.c_str(), nullptr, nullptr); // blocking
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

int VideoTexture::initOutputFrame() {
    frameRGB = av_frame_alloc();

    int numBytes = av_image_get_buffer_size(AV_PIX_FMT_RGB24, width, height, 1);
    buffer = (uint8_t*)av_malloc(numBytes * sizeof(uint8_t));

    av_image_fill_arrays(frameRGB->data, frameRGB->linesize, buffer, AV_PIX_FMT_RGB24, width, height, 1);

    swsContext = sws_getContext(inputCodecContext->width, inputCodecContext->height, inputCodecContext->pix_fmt,
                                width, height, AV_PIX_FMT_RGB24, SWS_BILINEAR, nullptr, nullptr, nullptr);

    return 0;
}

void VideoTexture::receiveVideo() {
    int ret = initFFMpeg();
    if (ret < 0) {
        return;
    }

    ret = initOutputFrame();
    if (ret < 0) {
        return;
    }

    videoReady = true;

    uint64_t prevTime = av_gettime();

    unsigned int poseId = -1;
    AVFrame* frame = av_frame_alloc();
    while (videoReady) {
        uint64_t receiveFrameStartTime = av_gettime();

        // read frame from URL
        int ret = av_read_frame(inputFormatContext, &packet);
        if (ret < 0) {
            av_log(nullptr, AV_LOG_ERROR, "Error reading frame: %s\n", av_err2str(ret));
            return;
        }

        stats.timeToReceiveFrame = (av_gettime() - receiveFrameStartTime) / MICROSECONDS_IN_SECOND;

        poseId = packet.pts;

        if (packet.stream_index == videoStreamIndex) {
            /* Decode received frame */
            {
                uint64_t decodeStartTime = av_gettime();

                // send packet to decoder
                ret = avcodec_send_packet(inputCodecContext, &packet);
                if (ret < 0 || ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    av_log(nullptr, AV_LOG_ERROR, "Error: Could not send packet to input decoder: %s\n", av_err2str(ret));
                    return;
                }

                // get frame from decoder
                ret = avcodec_receive_frame(inputCodecContext, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    av_packet_unref(&packet);
                    continue;
                }
                else if (ret < 0) {
                    av_log(nullptr, AV_LOG_ERROR, "Error: Could not receive raw frame from input decoder: %s\n", av_err2str(ret));
                    return;
                }

                stats.timeToDecode = (av_gettime() - decodeStartTime) / MICROSECONDS_IN_SECOND;
            }

            /* Resize video frame to fit output texture size */\
            {
                uint64_t resizeStartTime = av_gettime();

                frameRGBMutex.lock();
                frameRGB->opaque = reinterpret_cast<void*>(poseId);
                sws_scale(swsContext, (uint8_t const* const*)frame->data, frame->linesize,
                        0, inputCodecContext->height, frameRGB->data, frameRGB->linesize);
                frameRGBMutex.unlock();

                stats.timeToResize = (av_gettime() - resizeStartTime) / MICROSECONDS_IN_SECOND;
            }

            uint64_t elapsedTime = (av_gettime() - prevTime);
            stats.totalTimeToReceiveFrame = elapsedTime / MICROSECONDS_IN_SECOND;
            frameReceived++;

            prevTime = av_gettime();
        }
    }
}

uint32_t VideoTexture::draw() {
    if (!videoReady) {
        return -1;
    }

    frameRGBMutex.lock();
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, frameRGB->data[0]);
    frameRGBMutex.unlock();

    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(frameRGB->opaque));
}

void VideoTexture::cleanup() {
    videoReceiverThread.join();
    Texture::cleanup();
    avformat_close_input(&inputFormatContext);
    avformat_free_context(inputFormatContext);
}

