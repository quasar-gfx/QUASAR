#include <VideoTexture.h>

#undef av_err2str
#define av_err2str(errnum) av_make_error_string((char*)__builtin_alloca(AV_ERROR_MAX_STRING_SIZE), AV_ERROR_MAX_STRING_SIZE, errnum)

static int interrupt_callback(void* ctx) {
    bool* shouldTerminatePtr = (bool*)ctx;
    bool shouldTerminate = (shouldTerminatePtr != nullptr) ? *shouldTerminatePtr : false;
    return shouldTerminate;
}

VideoTexture::VideoTexture(const TextureCreateParams &params, const std::string &videoURL)
        : Texture(params)
        , videoURL("udp://" + videoURL + "?overrun_nonfatal=1&fifo_size=50000000") {
    videoReceiverThread = std::thread(&VideoTexture::receiveVideo, this);
}

int VideoTexture::initFFMpeg() {
    AVStream* inputVideoStream = nullptr;

    std::cout << "Waiting to receive video..." << std::endl;

    inputFormatCtx->interrupt_callback.callback = interrupt_callback;
    inputFormatCtx->interrupt_callback.opaque = &shouldTerminate;

    /* Setup input (to read video from url) */
    int ret = avformat_open_input(&inputFormatCtx, videoURL.c_str(), nullptr, nullptr); // blocking
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
        if (inputFormatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            videoStreamIndex = i;
            inputVideoStream = inputFormatCtx->streams[i];
            break;
        }
    }

    if (videoStreamIndex == -1 || inputVideoStream == nullptr) {
        av_log(nullptr, AV_LOG_ERROR, "No video stream found in the input URL.\n");
        return AVERROR_STREAM_NOT_FOUND;
    }

    /* Setup codec to decode input (video from URL) */
    auto codecID = inputVideoStream->codecpar->codec_id;
    auto inputCodec = avcodec_find_decoder(codecID);
    if (!inputCodec) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't allocate decoder.\n");
        return -1;
    }

    codecCtx = avcodec_alloc_context3(inputCodec);
    if (!codecCtx) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't allocate codec Ctx.\n");
        return -1;
    }

    ret = avcodec_parameters_to_context(codecCtx, inputVideoStream->codecpar);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't copy codec parameters to Ctx: %s\n", av_err2str(ret));
        return ret;
    }

    ret = avcodec_open2(codecCtx, inputCodec, nullptr);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't open codec: %s\n", av_err2str(ret));
        return ret;
    }

    swsCtx = sws_getContext(codecCtx->width, codecCtx->height, codecCtx->pix_fmt,
                            width, height, openglPixelFormat,
                            SWS_BILINEAR, nullptr, nullptr, nullptr);

    return 0;
}

void VideoTexture::receiveVideo() {
    int ret = initFFMpeg();
    if (ret < 0) {
        return;
    }

    videoReady = true;

    int prevTime = timeutils::getCurrTimeMillis();

    size_t bytesReceived = 0;
    pose_id_t poseID = -1;
    while (videoReady) {
        int receiveFrameStartTime = timeutils::getCurrTimeMillis();

        // read frame from URL
        int ret = av_read_frame(inputFormatCtx, packet);
        if (ret < 0) {
            av_log(nullptr, AV_LOG_ERROR, "Error reading frame: %s\n", av_err2str(ret));
            return;
        }

        stats.timeToReceiveMs = (timeutils::getCurrTimeMillis() - receiveFrameStartTime);

        if (packet->stream_index != videoStreamIndex) {
            continue;
        }

        bytesReceived = packet->size;

        // extract poseID from packet
        memcpy(&poseID, packet->data + packet->size - sizeof(pose_id_t), sizeof(pose_id_t));

        // remove poseID from packet
        packet->size -= sizeof(pose_id_t);

        /* Decode received frame */
        {
            int decodeStartTime = timeutils::getCurrTimeMillis();

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

            stats.timeToDecodeMs = (timeutils::getCurrTimeMillis() - decodeStartTime);
        }

        /* Resize video frame to fit output texture size */
        {
            int resizeStartTime = timeutils::getCurrTimeMillis();

            AVFrame* frameRGB = av_frame_alloc();

            int numBytes = av_image_get_buffer_size(openglPixelFormat, width, height, 1);
            uint8_t* buffer = (uint8_t*)av_malloc(numBytes * sizeof(uint8_t));
            av_image_fill_arrays(frameRGB->data, frameRGB->linesize, buffer, openglPixelFormat, width, height, 1);

            sws_scale(swsCtx, (uint8_t const* const*)frame->data, frame->linesize, 0, codecCtx->height, frameRGB->data, frameRGB->linesize);

            std::unique_lock<std::mutex> lock(m);

            FrameData frameData = {poseID, frameRGB, buffer};
            frames.push_back(frameData);
            if (frames.size() > maxQueueSize) {
                FrameData frameToFree = frames.front();
                frameToFree.free();
                frames.pop_front();
            }

            m.unlock();

            stats.timeToResizeMs = (timeutils::getCurrTimeMillis() - resizeStartTime);
        }

        int elapsedTimeMs = (timeutils::getCurrTimeMillis() - prevTime);
        stats.totalTimeToReceiveMs = elapsedTimeMs;
        framesReceived++;

        stats.bitrateMbps = ((bytesReceived * 8) / (stats.totalTimeToReceiveMs * MILLISECONDS_IN_SECOND));

        prevTime = timeutils::getCurrTimeMillis();
    }
}

pose_id_t VideoTexture::draw(pose_id_t poseID) {
    std::lock_guard<std::mutex> lock(m);
    if (!videoReady) {
        return -1;
    }

    if (frames.empty()) {
        return -1;
    }

    pose_id_t res = -1;
    AVFrame* frameRGB = nullptr;
    if (poseID == -1) {
        FrameData frameData = frames.back();
        frameRGB = frameData.frame;
        res = frameData.poseID;
    }
    else {
        for (auto it = frames.begin(); it != frames.end(); it++) {
            FrameData frameData = *it;
            if (frameData.poseID == poseID) {
                frameRGB = frameData.frame;
                res = frameData.poseID;
                break;
            }
        }
    }

    if (frameRGB == nullptr) {
        return -1;
    }

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, frameRGB->data[0]);

    return res;
}

pose_id_t VideoTexture::getLatestPoseID() {
    std::lock_guard<std::mutex> lock(m);
    if (frames.empty()) {
        return -1;
    }

    pose_id_t poseID;

    FrameData frameData = frames.back();
    poseID = frameData.poseID;
    return poseID;
}

void VideoTexture::cleanup() {
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

    Texture::cleanup();
}

