#include <cstring>

#include <spdlog/spdlog.h>

#include <Utils/FileIO.h>
#include <VideoTexture.h>

#undef av_err2str
#define av_err2str(errnum) av_make_error_string((char*)__builtin_alloca(AV_ERROR_MAX_STRING_SIZE), AV_ERROR_MAX_STRING_SIZE, errnum)

using namespace quasar;

static int interrupt_callback(void* ctx) {
    bool* shouldTerminatePtr = (bool*)ctx;
    bool shouldTerminate = (shouldTerminatePtr != nullptr) ? *shouldTerminatePtr : false;
    return shouldTerminate;
}

VideoTexture::VideoTexture(const TextureDataCreateParams& params,
                           const std::string& videoURL,
                           const std::string& formatName)
        : formatName(formatName)
        , Texture(params) {
    std::string sdpFileName = "stream.sdp";
#if defined(__ANDROID__)
    if (formatName != "mpegts") {
        sdpFileName = FileIO::copyFileToCache(sdpFileName);
        spdlog::info("Copied SDP file to: {}", sdpFileName);
    }
#else
    sdpFileName = "../assets/" + sdpFileName;
#endif

    this->videoURL = (formatName == "mpegts") ?
                        "udp://" + videoURL + "?overrun_nonfatal=1&fifo_size=50000000" :
                            sdpFileName;

    spdlog::info("Created VideoTexture that recvs from URL: {} ({})", videoURL, formatName);

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

    // Av_free(buffer);

    while (!frames.empty()) {
        frames.pop_front();
    }
}

int VideoTexture::initFFMpeg() {
    AVStream* inputVideoStream = nullptr;

    spdlog::info("Waiting to receive video...");

    inputFormatCtx->interrupt_callback.callback = interrupt_callback;
    inputFormatCtx->interrupt_callback.opaque = &shouldTerminate;

    AVDictionary* options = nullptr;
    av_dict_set(&options, "protocol_whitelist", "file,udp,rtp", 0);
    av_dict_set(&options, "fflags", "nobuffer", 0);
    // Av_dict_set(&options, "buffer_size", "1000000", 0);
    // Av_dict_set(&options, "max_delay", "500000", 0);

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

    // Find the video stream index
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
    spdlog::info("Decoder: {}", codec->name);

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
    spdlog::info("Video resolution: {}x{}", videoWidth, videoHeight);

    internalWidth = videoWidth;
    internalHeight = videoHeight;

    swsCtx = sws_getContext(videoWidth, videoHeight, codecCtx->pix_fmt,
                            internalWidth, internalHeight, openglPixelFormat,
                            SWS_BILINEAR, nullptr, nullptr, nullptr);

    return 0;
}

pose_id_t VideoTexture::unpackPoseIDFromFrame(AVFrame* frame) {
    // Extract poseID from frame
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

    uint64_t prevTime = timeutils::getTimeMicros();

    size_t bytesReceived = 0;
    while (videoReady) {
        time_t startRecvTime = timeutils::getTimeMicros();

        // Read frame from URL
        int ret = av_read_frame(inputFormatCtx, packet);
        if (ret < 0) {
            av_log(nullptr, AV_LOG_ERROR, "Error reading frame: %s\n", av_err2str(ret));
            return;
        }

        stats.timeToReceiveMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startRecvTime);

        if (packet->stream_index != videoStreamIndex) {
            continue;
        }

        bytesReceived = packet->size;

        /* Decode received frame */
        {
            time_t startDecodeTime = timeutils::getTimeMicros();

            // Send packet to decoder
            ret = avcodec_send_packet(codecCtx, packet);
            if (ret < 0 || ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                av_packet_unref(packet);
                av_log(nullptr, AV_LOG_ERROR, "Error: Could not send packet to input decoder: %s\n", av_err2str(ret));
                return;
            }

            av_packet_unref(packet);

            // Get frame from decoder
            ret = avcodec_receive_frame(codecCtx, frame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                continue;
            }
            else if (ret < 0) {
                av_log(nullptr, AV_LOG_ERROR, "Error: Could not receive raw frame from input decoder: %s\n", av_err2str(ret));
                return;
            }

            stats.timeToDecodeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startDecodeTime);
        }

        /* Resize video frame to fit output texture size */
        {
            time_t startResizeTime = timeutils::getTimeMicros();

            AVFrame* frameRGB = av_frame_alloc();

            int numBytes = av_image_get_buffer_size(openglPixelFormat, internalWidth, internalHeight, 1);
            uint8_t* imageBuffer = (uint8_t*)av_malloc(numBytes);
            av_image_fill_arrays(frameRGB->data, frameRGB->linesize, imageBuffer,
                                 openglPixelFormat, internalWidth, internalHeight, 1);

            sws_scale(swsCtx, (uint8_t const* const*)frame->data, frame->linesize,
                      0, videoHeight, frameRGB->data, frameRGB->linesize);

            pose_id_t poseID = unpackPoseIDFromFrame(frameRGB);

            // Copy buffer
            std::vector<char> bufferCopy((char*)imageBuffer, (char*)imageBuffer + numBytes);
            FrameData frameData = {poseID, std::move(bufferCopy)};
            {
                std::unique_lock<std::mutex> lock(m);

                frames.push_back(std::move(frameData));
                if (frames.size() > maxQueueSize) {
                    frames.pop_front();
                }
            }

            // Clean up temporary AVFrame and raw buffer
            av_free(imageBuffer);
            av_frame_free(&frameRGB);

            stats.timeToResizeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startResizeTime);
        }

        stats.totalTimeToReceiveMs = timeutils::microsToMillis(timeutils::getTimeMicros() - prevTime);
        framesReceived++;

        stats.bitrateMbps = ((bytesReceived * 8) / timeutils::millisToSeconds(stats.totalTimeToReceiveMs)) / BYTES_IN_MB;

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

    FrameData& resFrameData = frames.back();
    if (poseID != -1) { // search for frame with poseID
        for (auto& frameData : frames) {
            if (frameData.poseID == poseID) {
                resFrameData = frameData;
                break;
            }
        }
    }

    int stride = internalWidth;
    glPixelStorei(GL_UNPACK_ROW_LENGTH, stride);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, resFrameData.buffer.data());

    prevPoseID = resFrameData.poseID;

    return resFrameData.poseID;
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

void VideoTexture::resize(uint width, uint height) {
    internalWidth = width + poseIDOffset;
    internalHeight = height;
    Texture::resize(width, height);
}
