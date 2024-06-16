#include <glad/glad.h>

#include <VideoTexture.h>

#undef av_err2str
#define av_err2str(errnum) av_make_error_string((char*)__builtin_alloca(AV_ERROR_MAX_STRING_SIZE), AV_ERROR_MAX_STRING_SIZE, errnum)

VideoTexture::VideoTexture(const TextureCreateParams &params, const std::string &videoURL)
        : Texture(params)
        , width(params.width)
        , height(params.height)
        , videoURL("udp://" + videoURL + "?overrun_nonfatal=1&fifo_size=50000000") {
    videoReceiverThread = std::thread(&VideoTexture::receiveVideo, this);
}

int VideoTexture::initFFMpeg() {
    AVStream* inputVideoStream = nullptr;

    std::cout << "Waiting to receive video..." << std::endl;

    /* Setup input (to read video from url) */
    int ret = avformat_open_input(&inputFormatContext, videoURL.c_str(), nullptr, nullptr); // blocking
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Cannot open input URL: %s\n", av_err2str(ret));
        return ret;
    }

    ret = avformat_find_stream_info(inputFormatContext, nullptr);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Cannot find stream info: %s\n", av_err2str(ret));
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

    /* Setup codec to decode input (video from URL) */
    auto codecID = inputVideoStream->codecpar->codec_id;
    auto inputCodec = avcodec_find_decoder(codecID);
    if (!inputCodec) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't allocate decoder.\n");
        return -1;
    }

    codecContext = avcodec_alloc_context3(inputCodec);
    if (!codecContext) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't allocate codec context.\n");
        return -1;
    }

    ret = avcodec_parameters_to_context(codecContext, inputVideoStream->codecpar);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't copy codec parameters to context: %s\n", av_err2str(ret));
        return ret;
    }

    ret = avcodec_open2(codecContext, inputCodec, nullptr);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't open codec: %s\n", av_err2str(ret));
        return ret;
    }

    swsContext = sws_getContext(codecContext->width, codecContext->height, codecContext->pix_fmt,
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

    uint64_t prevTime = av_gettime();

    pose_id_t poseID = -1;
    while (videoReady) {
        uint64_t receiveFrameStartTime = av_gettime();

        // read frame from URL
        int ret = av_read_frame(inputFormatContext, packet);
        if (ret < 0) {
            av_log(nullptr, AV_LOG_ERROR, "Error reading frame: %s\n", av_err2str(ret));
            return;
        }

        stats.timeToReceiveFrame = (av_gettime() - receiveFrameStartTime) / MICROSECONDS_IN_SECOND;

        if (packet->stream_index != videoStreamIndex) {
            continue;
        }

        poseID = packet->pts;

        /* Decode received frame */
        {
            uint64_t decodeStartTime = av_gettime();

            // send packet to decoder
            ret = avcodec_send_packet(codecContext, packet);
            if (ret < 0 || ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                av_log(nullptr, AV_LOG_ERROR, "Error: Could not send packet to input decoder: %s\n", av_err2str(ret));
                return;
            }

            // get frame from decoder
            ret = avcodec_receive_frame(codecContext, frame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                continue;
            }
            else if (ret < 0) {
                av_log(nullptr, AV_LOG_ERROR, "Error: Could not receive raw frame from input decoder: %s\n", av_err2str(ret));
                return;
            }

            stats.timeToDecode = (av_gettime() - decodeStartTime) / MICROSECONDS_IN_SECOND;
        }

        /* Resize video frame to fit output texture size */
        {
            uint64_t resizeStartTime = av_gettime();

            AVFrame* frameRGB = av_frame_alloc();
            int numBytes = av_image_get_buffer_size(openglPixelFormat, width, height, 1);
            uint8_t* buffer = (uint8_t*)av_malloc(numBytes * sizeof(uint8_t));
            av_image_fill_arrays(frameRGB->data, frameRGB->linesize, buffer, openglPixelFormat, width, height, 1);

            frameRGB->opaque = reinterpret_cast<void*>(poseID);

            sws_scale(swsContext, (uint8_t const* const*)frame->data, frame->linesize, 0, codecContext->height, frameRGB->data, frameRGB->linesize);

            framesMutex.lock();

            frames.push_back(frameRGB);
            if (frames.size() > maxQueueSize) {
                av_frame_free(&frames.front());
                frames.erase(frames.begin());
            }

            framesMutex.unlock();

            stats.timeToResize = (av_gettime() - resizeStartTime) / MICROSECONDS_IN_SECOND;
        }

        uint64_t elapsedTime = (av_gettime() - prevTime);
        stats.totalTimeToReceiveFrame = elapsedTime / MICROSECONDS_IN_SECOND;
        framesReceived++;

        prevTime = av_gettime();
    }
}

pose_id_t VideoTexture::draw(pose_id_t poseID) {
    if (!videoReady) {
        return -1;
    }

    if (frames.empty()) {
        return -1;
    }

    framesMutex.lock();

    AVFrame* frameRGB = nullptr;
    if (poseID == -1) {
        frameRGB = frames.back();
    }
    else {
        for (auto it = frames.begin(); it != frames.end(); it++) {
            if (reinterpret_cast<uintptr_t>((*it)->opaque) == poseID) {
                frameRGB = *it;
                break;
            }
        }
    }

    framesMutex.unlock();

    if (frameRGB == nullptr) {
        return -1;
    }

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, frameRGB->data[0]);

    return static_cast<pose_id_t>(reinterpret_cast<uintptr_t>(frameRGB->opaque));
}

bool VideoTexture::hasPoseID(pose_id_t poseID) {
    if (frames.empty()) {
        return false;
    }

    for (auto it = frames.begin(); it != frames.end(); it++) {
        if (reinterpret_cast<uintptr_t>((*it)->opaque) == poseID) {
            return true;
        }
    }

    return false;
}

pose_id_t VideoTexture::getLatestPoseID() {
    if (frames.empty()) {
        return -1;
    }

    pose_id_t poseID;

    framesMutex.lock();
    AVFrame* frameRGB = frames.front();
    poseID = static_cast<pose_id_t>(reinterpret_cast<uintptr_t>(frameRGB->opaque));
    framesMutex.unlock();

    return poseID;
}

void VideoTexture::cleanup() {
    videoReceiverThread.join();
    Texture::cleanup();
    avformat_close_input(&inputFormatContext);
    avformat_free_context(inputFormatContext);
    avcodec_free_context(&codecContext);
    sws_freeContext(swsContext);
    av_frame_free(&frame);
    av_packet_free(&packet);
    for (auto it = frames.begin(); it != frames.end(); it++) {
        av_frame_free(&(*it));
    }
    frames.clear();
}

