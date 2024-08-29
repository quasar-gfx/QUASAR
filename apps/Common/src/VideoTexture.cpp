#include <iostream>
#include <cstring>

#include <VideoTexture.h>

#undef av_err2str
#define av_err2str(errnum) av_make_error_string((char*)__builtin_alloca(AV_ERROR_MAX_STRING_SIZE), AV_ERROR_MAX_STRING_SIZE, errnum)

static int interrupt_callback(void* ctx) {
    bool* shouldTerminatePtr = (bool*)ctx;
    bool shouldTerminate = (shouldTerminatePtr != nullptr) ? *shouldTerminatePtr : false;
    return shouldTerminate;
}

VideoTexture::VideoTexture(const TextureDataCreateParams &params, const std::string &videoURL)
        : Texture(params)
        , videoURL("udp://" + videoURL + "?overrun_nonfatal=1&fifo_size=50000000") {
    videoReceiverThread = std::thread(&VideoTexture::receiveVideo, this);
}

VideoTexture::~VideoTexture() {
    shouldTerminate = true;
    videoReady = false;

    if (videoReceiverThread.joinable()) {
        videoReceiverThread.join();
    }

    avcodec_free_context(&codecCtx);

    avformat_close_input(&inputFormatCtx);
    avformat_free_context(inputFormatCtx);

    av_frame_free(&frame);
    av_packet_unref(packet);
    av_packet_free(&packet);

    // av_free(buffer);

    while (!frames.empty()) {
        FrameData frameData = frames.front();
        frameData.free();
        frames.pop_front();
    }
}



int VideoTexture::initFFMpeg() {
    AVStream* inputVideoStream = nullptr;

    std::cout << "Waiting to receive video..." << std::endl;

    inputFormatCtx->interrupt_callback.callback = interrupt_callback;
    inputFormatCtx->interrupt_callback.opaque = &shouldTerminate;

    AVDictionary* options = nullptr;
    av_dict_set(&options, "protocol_whitelist", "file,udp,rtp", 0);
    // av_dict_set(&options, "buffer_size", "1000k", 0);
    // av_dict_set(&options, "max_delay", "500k", 0);

    /* Setup input (to read video from url) */
    int ret = avformat_open_input(&inputFormatCtx, videoURL.c_str(), nullptr, &options); // blocking
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Cannot open input URL: %s\n", av_err2str(ret));
        return ret;
    }

    ret = avformat_find_stream_info(inputFormatCtx, nullptr);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Cannot find stream info: %s\n", av_err2str(ret));
        return ret;
    }

    // find the video stream index
    for (int i = 0; i < inputFormatCtx->nb_streams; i++) {
        AVStream *stream = inputFormatCtx->streams[i];
        AVCodecParameters *codec_params = stream->codecpar;
        if (codec_params->codec_type == AVMEDIA_TYPE_VIDEO) {
            videoStreamIndex = i;
            inputVideoStream = stream;
            break;
        }
    }

    if (videoStreamIndex == -1 || inputVideoStream == nullptr) {
        av_log(nullptr, AV_LOG_ERROR, "No video stream found in the input URL.\n");
        return AVERROR_STREAM_NOT_FOUND;
    }

    /* Setup codec to decode input (video from URL) */
#if defined(__linux__) && !defined(__ANDROID__)
    std::string decoderName = "h264_cuvid";
    auto codec = avcodec_find_decoder_by_name(decoderName.c_str());
#else
    auto codec = avcodec_find_decoder(inputVideoStream->codecpar->codec_id);
#endif
    if (!codec) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't allocate decoder.\n");
        return -1;
    }
    std::cout << "Decoder: " << codec->name << std::endl;

    codecCtx = avcodec_alloc_context3(codec);
    if (!codecCtx) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't allocate codec Ctx.\n");
        return -1;
    }

    ret = avcodec_parameters_to_context(codecCtx, inputVideoStream->codecpar);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't copy codec parameters to Ctx: %s\n", av_err2str(ret));
        return ret;
    }

    ret = avcodec_open2(codecCtx, codec, nullptr);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't open codec: %s\n", av_err2str(ret));
        return ret;
    }

    videoWidth = codecCtx->width;
    videoHeight = codecCtx->height;
    std::cout << "Video resolution: " << videoWidth << "x" << videoHeight << std::endl;

    internalWidth = videoWidth;
    internalHeight = videoHeight;

    swsCtx = sws_getContext(videoWidth, videoHeight, codecCtx->pix_fmt,
                            internalWidth, internalHeight, openglPixelFormat,
                            SWS_BILINEAR, nullptr, nullptr, nullptr);

    return 0;
}

pose_id_t VideoTexture::unpackPoseIDFromFrame(AVFrame* frame) {
    // extract poseID from frame
    const int numVotes = 32;
    pose_id_t poseID = 0;
    for (int i = 0; i < poseIDOffset; i++) {
        int votes = 0;
        for (int j = 0; j < numVotes; j++) {
            int index = j * internalWidth * 3 + (internalWidth - 1 - i) * 3;
            uint8_t value = frame->data[0][index];
            if (value > 127) {
                votes++;
            }
        }
        poseID |= (votes > numVotes / 2) << i;
    }

    return poseID;
}

void VideoTexture::receiveVideo() {
    int ret = initFFMpeg();
    if (ret < 0) {
        return;
    }

    videoReady = true;

    float prevTime = timeutils::getTimeMicros();

    size_t bytesReceived = 0;
    pose_id_t poseID = -1;
    while (videoReady) {
        int receiveFrameStartTime = timeutils::getTimeMicros();

        // read frame from URL
        int ret = av_read_frame(inputFormatCtx, packet);
        if (ret < 0) {
            av_log(nullptr, AV_LOG_ERROR, "Error reading frame: %s\n", av_err2str(ret));
            return;
        }

        stats.timeToReceiveMs = timeutils::microsToMillis(timeutils::getTimeMicros() - receiveFrameStartTime);

        if (packet->stream_index != videoStreamIndex) {
            continue;
        }

        bytesReceived = packet->size;

        /* Decode received frame */
        {
            int decodeStartTime = timeutils::getTimeMicros();

            // send packet to decoder
            ret = avcodec_send_packet(codecCtx, packet);
            if (ret < 0 || ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                av_packet_unref(packet);
                av_log(nullptr, AV_LOG_ERROR, "Error: Could not send packet to input decoder: %s\n", av_err2str(ret));
                return;
            }

            av_packet_unref(packet);

            // get frame from decoder
            ret = avcodec_receive_frame(codecCtx, frame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                continue;
            }
            else if (ret < 0) {
                av_log(nullptr, AV_LOG_ERROR, "Error: Could not receive raw frame from input decoder: %s\n", av_err2str(ret));
                return;
            }

            stats.timeToDecodeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - decodeStartTime);
        }

        /* Resize video frame to fit output texture size */
        {
            int resizeStartTime = timeutils::getTimeMicros();

            AVFrame* frameRGB = av_frame_alloc();

            int numBytes = av_image_get_buffer_size(openglPixelFormat, internalWidth, internalHeight, 1);
            uint8_t* buffer = (uint8_t*)av_malloc(numBytes * sizeof(uint8_t));
            av_image_fill_arrays(frameRGB->data, frameRGB->linesize, buffer, openglPixelFormat, internalWidth, internalHeight, 1);

            sws_scale(swsCtx, (uint8_t const* const*)frame->data, frame->linesize, 0, videoHeight, frameRGB->data, frameRGB->linesize);

            poseID = unpackPoseIDFromFrame(frameRGB);

            {
                std::unique_lock<std::mutex> lock(m);

                FrameData frameData = {poseID, frameRGB, buffer};
                frames.push_back(frameData);
                if (frames.size() > maxQueueSize) {
                    FrameData frameToFree = frames.front();
                    frameToFree.free();
                    frames.pop_front();
                }
            }

            stats.timeToResizeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - resizeStartTime);
        }

        stats.totalTimeToReceiveMs = timeutils::microsToMillis(timeutils::getTimeMicros() - prevTime);
        framesReceived++;

        stats.bitrateMbps = ((bytesReceived * 8) / timeutils::millisToSeconds(stats.totalTimeToReceiveMs)) / MBPS_TO_BPS;

        prevTime = timeutils::getTimeMicros();
    }
}

pose_id_t VideoTexture::draw(pose_id_t poseID) {
    std::lock_guard<std::mutex> lock(m);
    if (!videoReady) {
        return -1;
    }

    if (frames.empty()) {
        return prevPoseID;
    }

    pose_id_t resPoseID = -1;
    AVFrame* frameRGB = nullptr;
    if (poseID == -1) {
        FrameData frameData = frames.back();
        frameRGB = frameData.frame;
        resPoseID = frameData.poseID;
    }
    else {
        for (auto it = frames.begin(); it != frames.end(); it++) {
            FrameData frameData = *it;
            if (frameData.poseID == poseID) {
                frameRGB = frameData.frame;
                resPoseID = frameData.poseID;
                break;
            }
        }

        if (frameRGB == nullptr) {
            return -1;
        }
    }

    int stride = internalWidth;
    glPixelStorei(GL_UNPACK_ROW_LENGTH, stride);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, frameRGB->data[0]);

    prevPoseID = resPoseID;

    return resPoseID;
}

pose_id_t VideoTexture::getLatestPoseID() {
    std::lock_guard<std::mutex> lock(m);
    if (frames.empty()) {
        return -1;
    }

    FrameData frameData = frames.back();
    pose_id_t poseID = frameData.poseID;
    return poseID;
}

void VideoTexture::resize(unsigned int width, unsigned int height) {
    internalWidth = width + poseIDOffset;
    internalHeight = height;
    Texture::resize(width, height);
}
