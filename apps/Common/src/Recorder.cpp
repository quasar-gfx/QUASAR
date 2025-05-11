#include <Recorder.h>
#include <Utils/FileIO.h>
#include <Utils/TimeUtils.h>

#undef av_err2str
#define av_err2str(errnum) av_make_error_string((char*)__builtin_alloca(AV_ERROR_MAX_STRING_SIZE), AV_ERROR_MAX_STRING_SIZE, errnum)

using namespace quasar;

Recorder::~Recorder() {
    if (running) {
        stop();
    }
}

void Recorder::setOutputPath(const std::string& path) {
    // add / if not present
    outputPath = path;
    if (outputPath.back() != '/') {
        outputPath += "/";
    }
    // create output path if it doesn't exist
    if (!std::filesystem::exists(outputPath)) {
        std::filesystem::create_directories(outputPath);
    }
}

void Recorder::setTargetFrameRate(int targetFrameRate) {
    targetFrameRate = std::max(1, targetFrameRate);
    lastCaptureTime = timeutils::getTimeMillis();
    frameCount = 0;
}

void Recorder::saveScreenshotToFile(const std::string &fileName, bool saveAsHDR) {
    effect.drawToRenderTarget(renderer, *this);

    if (saveAsHDR) {
        saveColorAsHDR(outputPath + fileName + ".hdr");
    }
    else {
        saveColorAsPNG(outputPath + fileName + ".png");
    }
}

void Recorder::start() {
    running = true;
    frameCount = 0;

    recordingStartTime = timeutils::getTimeMillis();
    lastCaptureTime = recordingStartTime;

    std::ofstream pathFile(outputPath + "camera_path.txt");
    pathFile.close();

    if (outputFormat == OutputFormat::MP4) {
        initializeFFmpeg();
    }

    saveThreadPool = std::make_unique<BS::thread_pool<>>(numThreads);
    for (int i = 0; i < numThreads; i++) {
        auto future = saveThreadPool->submit_task([this, i]() {
            saveFrames(i);
        });
        (void)future;
    }
}

void Recorder::stop() {
    if (!running) {
        return;
    }
    running = false;

    queueCV.notify_all();
    saveThreadPool->wait();
    saveThreadPool.reset();

    if (outputFormat == OutputFormat::MP4) {
        finalizeFFmpeg();
    }

    frameQueue = std::queue<FrameData>();
    frameCount = 0;
}

void Recorder::captureFrame(const Camera &camera) {
    int64_t currentTime = timeutils::getTimeMillis();
    int64_t elapsedTime = currentTime - recordingStartTime;

    effect.drawToRenderTarget(renderer, *this);

    std::vector<uint8_t> renderTargetData(width * height * 4);

#if !defined(__APPLE__) && !defined(__ANDROID__)
    cudaImage.copyToArray(
        width * 4, height, width * 4, renderTargetData.data());
#else
    readPixels(renderTargetData.data());
#endif

    {
        std::lock_guard<std::mutex> lock(queueMutex);
        frameQueue.push(
            FrameData{
                frameCount, camera.getPosition(), camera.getRotationEuler(), std::move(renderTargetData), elapsedTime});
    }
    queueCV.notify_one();

    frameCount++;
    lastCaptureTime = currentTime;
}

void Recorder::saveFrames(int threadID) {
    /* setup frame */
    AVFrame* frame = av_frame_alloc();

    frame->width = width;
    frame->height = height;
    frame->format = videoPixelFormat;
    int ret = av_frame_get_buffer(frame, 0);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Could not allocate frame data: %s\n", av_err2str(ret));
        return;
    }

    ret = av_frame_make_writable(frame);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Could not make frame writable: %s\n", av_err2str(ret));
        return;
    }

    /* setup packet */
    AVPacket* packet = av_packet_alloc();
    ret = av_packet_make_writable(packet);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Could not make packet writable: %s\n", av_err2str(ret));
        return;
    }

    while (running || !frameQueue.empty()) {
        FrameData frameData;
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            queueCV.wait(lock, [this] { return !frameQueue.empty() || !running; });
            if (!running && frameQueue.empty()) {
                break;
            }

            frameData = std::move(frameQueue.front());
            frameQueue.pop();
        }

        int frameID = frameData.ID;
        auto &renderTargetData = frameData.data;

        if (outputFormat == OutputFormat::MP4) {
            for (int y = 0; y < height / 2; ++y) {
                for (int x = 0; x < width * 4; ++x) {
                    std::swap(renderTargetData[y * width * 4 + x],
                              renderTargetData[(height - 1 - y) * width * 4 + x]);
                }
            }

            {
                std::lock_guard<std::mutex> lock(swsMutex);

                const uint8_t* srcData[] = { renderTargetData.data() };
                int srcStride[] = { static_cast<int>(width * 4) }; // RGBA has 4 bytes per pixel

                sws_scale(swsCtx, srcData, srcStride, 0, height, frame->data, frame->linesize);
            }

            ret = avcodec_send_frame(codecCtx, frame);
            if (ret < 0 || ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                av_log(nullptr, AV_LOG_ERROR, "Error: Could not send frame to output encoder: %s\n", av_err2str(ret));
                continue;
            }

            ret = avcodec_receive_packet(codecCtx, packet);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                continue;
            }
            else if (ret < 0) {
                av_log(nullptr, AV_LOG_ERROR, "Error: Could not receive frame from encoder: %s\n", av_err2str(ret));
                break;
            }

            AVRational timeBase = outputFormatCtx->streams[outputVideoStream->index]->time_base;
            packet->pts = av_rescale_q(frameID, (AVRational){1, targetFrameRate}, timeBase);
            packet->dts = packet->pts;

            av_interleaved_write_frame(outputFormatCtx, packet);
            av_packet_unref(packet);
        }
        else {
            std::stringstream ss;
            ss << outputPath << "frame_" << std::setw(6) << std::setfill('0') << frameID;
            std::string fileName = ss.str();

            FileIO::flipVerticallyOnWrite(true);
            if (outputFormat == OutputFormat::PNG) {
                FileIO::saveAsPNG(fileName + ".png", width, height, 4, renderTargetData.data());
            }
            else {
                FileIO::saveAsJPG(fileName + ".jpg", width, height, 4, renderTargetData.data());
            }
        }

        {
            std::lock_guard<std::mutex> lock(cameraPathMutex);
            std::ofstream pathFile(outputPath + "camera_path.txt", std::ios::app);
            pathFile << std::fixed << std::setprecision(4)
                     << frameData.position.x << " "
                     << frameData.position.y << " "
                     << frameData.position.z << " "
                     << frameData.euler.x << " "
                     << frameData.euler.y << " "
                     << frameData.euler.z << " "
                     << frameData.elapsedTime << std::endl;
            pathFile.close();
        }
    }

    av_packet_free(&packet);
    av_frame_free(&frame);
}

void Recorder::initializeFFmpeg() {
#ifdef __APPLE__
    std::string encoderName = "h264_videotoolbox";
#elif __linux__
    std::string encoderName = "h264_nvenc";
#else
    std::string encoderName = "libx264";
#endif

    auto outputCodec = avcodec_find_encoder_by_name(encoderName.c_str());
    if (!outputCodec) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't allocate encoder.\n");
        throw std::runtime_error("Recorder could not be created.");
    }

    codecCtx = avcodec_alloc_context3(outputCodec);
    if (!codecCtx) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't allocate codec context.\n");
        throw std::runtime_error("Recorder could not be created.");
    }

    codecCtx->pix_fmt = videoPixelFormat;
    codecCtx->width = width;
    codecCtx->height = height;
    codecCtx->time_base = (AVRational){1, targetFrameRate};
    codecCtx->framerate = (AVRational){targetFrameRate, 1};
    codecCtx->bit_rate = 30 * BYTES_IN_MB;
    codecCtx->max_b_frames = 2;
    codecCtx->gop_size = 20;

    av_opt_set(codecCtx->priv_data, "crf", "18", 0);
    av_opt_set(codecCtx->priv_data, "preset", "slow", 0);
    av_opt_set(codecCtx->priv_data, "profile", "high", 0);

    int ret = avcodec_open2(codecCtx, outputCodec, nullptr);
    if (ret < 0) {
        throw std::runtime_error("Failed to open codec");
    }

    ret = avformat_alloc_output_context2(&outputFormatCtx, nullptr, nullptr, (outputPath + "output.mp4").c_str());
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Could not allocate output context: %s\n", av_err2str(ret));
        throw std::runtime_error("Recorder could not be created.");
    }

    outputVideoStream = avformat_new_stream(outputFormatCtx, nullptr);
    if (!outputVideoStream) {
        throw std::runtime_error("Failed to create new video stream");
    }

    outputVideoStream->time_base = codecCtx->time_base;

    avcodec_parameters_from_context(outputVideoStream->codecpar, codecCtx);

    if (!(outputFormatCtx->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&outputFormatCtx->pb, (outputPath + "output.mp4").c_str(), AVIO_FLAG_WRITE);
        if (ret < 0) {
            av_log(nullptr, AV_LOG_ERROR, "Cannot open output file\n");
            throw std::runtime_error("Recorder could not be created.");
        }
    }

    ret = avformat_write_header(outputFormatCtx, nullptr);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error writing header\n");
        throw std::runtime_error("Recorder could not be created.");
    }

    swsCtx = sws_getContext(width, height, rgbaPixelFormat,
                            width, height, videoPixelFormat,
                            SWS_BICUBIC, nullptr, nullptr, nullptr);
    if (!swsCtx) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Could not allocate conversion context\n");
        throw std::runtime_error("Recorder could not be created.");
    }
}

void Recorder::finalizeFFmpeg() {
    /* setup packet */
    AVPacket* packet = av_packet_alloc();
    int ret = av_packet_make_writable(packet);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Could not make packet writable: %s\n", av_err2str(ret));
        return;
    }

    // flush encoder
    avcodec_send_frame(codecCtx, nullptr);
    while (true) {
        ret = avcodec_receive_packet(codecCtx, packet);
        if (ret != 0) {
            break;
        }

        AVRational timeBase = outputFormatCtx->streams[outputVideoStream->index]->time_base;
        packet->pts = av_rescale_q(frameCount, (AVRational){1, targetFrameRate}, timeBase);
        packet->dts = packet->pts;

        av_interleaved_write_frame(outputFormatCtx, packet);
        av_packet_unref(packet);

        frameCount++;
    }
    av_packet_free(&packet);

    if (outputFormatCtx) {
        if (outputFormatCtx->pb != nullptr) {
            av_write_trailer(outputFormatCtx);
            if (!(outputFormatCtx->oformat->flags & AVFMT_NOFILE)) {
                avio_closep(&outputFormatCtx->pb);
            }
        }
    }

    if (swsCtx) {
        sws_freeContext(swsCtx);
        swsCtx = nullptr;
    }

    if (codecCtx) {
        avcodec_free_context(&codecCtx);
        codecCtx = nullptr;
    }

    if (outputFormatCtx) {
        avformat_free_context(outputFormatCtx);
        outputFormatCtx = nullptr;
    }

    frameCount = 0;
}

void Recorder::setFormat(OutputFormat format) {
    if (running) {
        return;
    }

    outputFormat = format;
}
