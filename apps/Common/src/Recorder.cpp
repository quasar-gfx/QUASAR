#include <Recorder.h>
#include <Utils/FileIO.h>
#include <Utils/TimeUtils.h>

#undef av_err2str
#define av_err2str(errnum) av_make_error_string((char*)__builtin_alloca(AV_ERROR_MAX_STRING_SIZE), AV_ERROR_MAX_STRING_SIZE, errnum)

using namespace quasar;

Recorder::Recorder(
        const RenderTargetCreateParams& params,
        OpenGLRenderer& renderer,
        PostProcessingEffect& effect,
        const Path& outputPath,
        int targetFrameRate,
        uint numThreads)
    : RenderTarget(params)
    , renderer(renderer)
    , effect(effect)
    , targetFrameRate(targetFrameRate)
    , numThreads(numThreads)
    , outputPath(outputPath)
    , outputFormats({"MP4", "PNG", "JPG"})
#if defined(HAS_CUDA)
    , cudaImage(colorTexture)
#endif
{
    setOutputPath(outputPath);
}

Recorder::~Recorder() {
    if (running) {
        stop();
    }
}

void Recorder::setOutputPath(const Path& path) {
    path.mkdir();
    outputPath = path;
}

void Recorder::setTargetFrameRate(int targetFrameRate) {
    targetFrameRate = std::max(1, targetFrameRate);
    frameCount = 0;
}

void Recorder::saveScreenshotToFile(const Path& filename, bool saveToHDR) {
    effect.drawToRenderTarget(renderer, *this);

    if (saveToHDR) {
        saveColorAsHDR(filename.withExtension(".hdr"));
    }
    else {
        saveColorAsPNG(filename.withExtension(".png"));
    }
}

void Recorder::start() {
    running = true;
    frameCount = 0;
    recordingStartTime = timeutils::getTimeMillis();

    if (outputFormat == OutputFormat::MP4) {
        initializeFFmpeg();
        saveThreadPool = std::make_unique<BS::thread_pool<>>(1);
        (void)saveThreadPool->submit_task([this]() { saveFrames(0); });
    }
    else {
        saveThreadPool = std::make_unique<BS::thread_pool<>>(numThreads);
        for (int i = 0; i < numThreads; i++) {
            (void)saveThreadPool->submit_task([this, i]() { saveFrames(i); });
        }
    }
}

void Recorder::stop() {
    if (!running) return;
    running = false;
    saveThreadPool->wait();
    saveThreadPool.reset();

    if (outputFormat == OutputFormat::MP4) {
        finalizeFFmpeg();
    }

    // Clear queue
    FrameData dummy;
    while (frameQueue.try_dequeue(dummy)) {}
    frameCount = 0;
}

void Recorder::captureFrame(const Camera& camera) {
    int64_t currentTime = timeutils::getTimeMillis();
    int64_t elapsedTime = currentTime - recordingStartTime;

    effect.drawToRenderTarget(renderer, *this);

    std::vector<uint8_t> renderTargetData(width * height * 4);

#if defined(HAS_CUDA)
    cudaImage.copyArrayToHost(width * 4, height, width * 4, renderTargetData.data());
#else
    readPixels(renderTargetData.data());
#endif

    frameQueue.enqueue(FrameData{
        frameCount,
        camera.getPosition(),
        camera.getRotationEuler(),
        std::move(renderTargetData),
        elapsedTime
    });

    frameCount++;
}

void Recorder::saveFrames(int threadID) {
    AVFrame* frame = nullptr;
    AVPacket* packet = nullptr;

    if (outputFormat == OutputFormat::MP4) {
        frame = av_frame_alloc();
        frame->width = width;
        frame->height = height;
        frame->format = videoPixelFormat;
        int ret = av_frame_get_buffer(frame, 0);
        if (ret < 0) return;
        ret = av_frame_make_writable(frame);
        if (ret < 0) return;

        packet = av_packet_alloc();
        ret = av_packet_make_writable(packet);
        if (ret < 0) return;
    }

    while (running || frameQueue.size_approx() > 0) {
        FrameData frameData;
        if (!frameQueue.try_dequeue(frameData)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        int frameID = frameData.ID;
        auto& renderTargetData = frameData.data;

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
                int srcStride[] = { static_cast<int>(width * 4) };
                sws_scale(swsCtx, srcData, srcStride, 0, height, frame->data, frame->linesize);
            }

            int ret = avcodec_send_frame(codecCtx, frame);
            if (ret < 0 || ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) continue;
            ret = avcodec_receive_packet(codecCtx, packet);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) continue;
            else if (ret < 0) break;

            AVRational timeBase = outputFormatCtx->streams[outputVideoStream->index]->time_base;
            packet->pts = av_rescale_q(frameID, {1, targetFrameRate}, timeBase);
            packet->dts = packet->pts;

            av_interleaved_write_frame(outputFormatCtx, packet);
            av_packet_unref(packet);
        }
        else {
            std::stringstream ss;
            ss << "frame_" << std::setw(6) << std::setfill('0') << frameID;
            Path fileNameBase = outputPath / ss.str();

            FileIO::flipVerticallyOnWrite(true);
            if (outputFormat == OutputFormat::PNG) {
                FileIO::saveToPNG(fileNameBase.withExtension(".png"), width, height, 4, renderTargetData.data());
            }
            else {
                FileIO::saveToJPG(fileNameBase.withExtension(".jpg"), width, height, 4, renderTargetData.data());
            }
        }

        {
            std::lock_guard<std::mutex> lock(cameraPathMutex);
            std::ostringstream pathFile;
            pathFile << std::fixed << std::setprecision(4)
                     << frameData.position.x << " " << frameData.position.y << " " << frameData.position.z << " "
                     << frameData.euler.x << " " << frameData.euler.y << " " << frameData.euler.z << " "
                     << frameData.elapsedTime << std::endl;
            FileIO::saveToTextFile(outputPath / "camera_path.txt", pathFile.str(), true);
        }
    }

    if (frame) av_frame_free(&frame);
    if (packet) av_packet_free(&packet);
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
    if (!outputCodec) throw std::runtime_error("Recorder could not be created.");

    codecCtx = avcodec_alloc_context3(outputCodec);
    if (!codecCtx) throw std::runtime_error("Recorder could not be created.");

    codecCtx->pix_fmt = videoPixelFormat;
    codecCtx->width = width;
    codecCtx->height = height;
    codecCtx->time_base = {1, targetFrameRate};
    codecCtx->framerate = {targetFrameRate, 1};
    codecCtx->bit_rate = targetBitRate * BYTES_PER_MEGABYTE;
    codecCtx->max_b_frames = max_b_frames;
    codecCtx->gop_size = gop_size;

    av_opt_set(codecCtx->priv_data, "preset", preset.c_str(), 0);
    av_opt_set(codecCtx->priv_data, "crf", crf.c_str(), 0);
    av_opt_set(codecCtx->priv_data, "profile", profile.c_str(), 0);

    int ret = avcodec_open2(codecCtx, outputCodec, nullptr);
    if (ret < 0) throw std::runtime_error("Failed to open codec");

    ret = avformat_alloc_output_context2(&outputFormatCtx, nullptr, nullptr, (outputPath / "output.mp4").c_str());
    if (ret < 0) throw std::runtime_error("Recorder could not be created.");

    outputVideoStream = avformat_new_stream(outputFormatCtx, nullptr);
    if (!outputVideoStream) throw std::runtime_error("Failed to create video stream");

    outputVideoStream->time_base = codecCtx->time_base;
    avcodec_parameters_from_context(outputVideoStream->codecpar, codecCtx);

    if (!(outputFormatCtx->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&outputFormatCtx->pb, (outputPath / "output.mp4").c_str(), AVIO_FLAG_WRITE);
        if (ret < 0) throw std::runtime_error("Failed to open output file");
    }

    ret = avformat_write_header(outputFormatCtx, nullptr);
    if (ret < 0) throw std::runtime_error("Failed to write header");

    swsCtx = sws_getContext(width, height, rgbaPixelFormat,
                            width, height, videoPixelFormat,
                            SWS_BICUBIC, nullptr, nullptr, nullptr);
    if (!swsCtx) throw std::runtime_error("Could not allocate conversion context");
}

void Recorder::finalizeFFmpeg() {
    AVPacket* packet = av_packet_alloc();
    av_packet_make_writable(packet);

    avcodec_send_frame(codecCtx, nullptr);
    while (true) {
        int ret = avcodec_receive_packet(codecCtx, packet);
        if (ret != 0) break;

        AVRational timeBase = outputFormatCtx->streams[outputVideoStream->index]->time_base;
        packet->pts = av_rescale_q(frameCount, {1, targetFrameRate}, timeBase);
        packet->dts = packet->pts;

        av_interleaved_write_frame(outputFormatCtx, packet);
        av_packet_unref(packet);

        frameCount++;
    }

    av_packet_free(&packet);
    av_write_trailer(outputFormatCtx);

    if (!(outputFormatCtx->oformat->flags & AVFMT_NOFILE)) {
        avio_closep(&outputFormatCtx->pb);
    }

    sws_freeContext(swsCtx);
    avcodec_free_context(&codecCtx);
    avformat_free_context(outputFormatCtx);

    codecCtx = nullptr;
    swsCtx = nullptr;
    outputFormatCtx = nullptr;
    frameCount = 0;
}

void Recorder::setFormat(OutputFormat format) {
    if (running) return;
    outputFormat = format;
}
