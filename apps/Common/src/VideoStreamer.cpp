#include <VideoStreamer.h>

#undef av_err2str
#define av_err2str(errnum) av_make_error_string((char*)__builtin_alloca(AV_ERROR_MAX_STRING_SIZE), AV_ERROR_MAX_STRING_SIZE, errnum)

static int interrupt_callback(void* ctx) {
    bool* shouldTerminatePtr = (bool*)ctx;
    bool shouldTerminate = (shouldTerminatePtr != nullptr) ? *shouldTerminatePtr : false;
    return shouldTerminate;
}

VideoStreamer::VideoStreamer(RenderTarget* renderTarget, const std::string &videoURL)
        : renderTarget(renderTarget)
        , width(renderTarget->width)
        , height(renderTarget->height)
        , videoURL("udp://" + videoURL) {
    int ret;

#ifndef __APPLE__
    ret = initCuda();
    if (ret < 0) {
        throw std::runtime_error("Error: Couldn't initialize CUDA");
    }

    /* init ffmpeg device */
    deviceCtx = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_CUDA);
    AVHWDeviceContext *hwdevCtx = (AVHWDeviceContext*)deviceCtx->data;
    auto cudaHwdevCtx = (AVCUDADeviceContext*)hwdevCtx->hwctx;

    CUcontext cuCtx;
    CUresult cuRes = cuCtxGetCurrent(&cuCtx);
    if (cuRes != CUDA_SUCCESS) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't get current CUDA context: %s\n", av_err2str(cuRes));
        throw std::runtime_error("Error: Couldn't get current CUDA context");
    }

    cudaHwdevCtx->cuda_ctx = cuCtx;
    cudaHwdevCtx->stream = nullptr;

    ret = av_hwdevice_ctx_init(deviceCtx);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't initialize CUDA device context: %s\n", av_err2str(ret));
        throw std::runtime_error("Error: Couldn't initialize CUDA device context");
    }

    /* init ffmpeg cuda device */
    ret = av_hwdevice_ctx_create(&cudaDeviceCtx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't create CUDA device context: %s\n", av_err2str(ret));
        throw std::runtime_error("Error: Couldn't create CUDA device context");
    }

    /* init ffmpeg frame context */
    frameCtx = av_hwframe_ctx_alloc(deviceCtx);
    if (!frameCtx) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't allocate frame context\n");
        throw std::runtime_error("Error: Couldn't allocate frame context");
    }

    auto hwframeCtx = reinterpret_cast<AVHWFramesContext*>(frameCtx->data);
    hwframeCtx->format = AV_PIX_FMT_CUDA;
    // hwframeCtx->sw_format = AV_PIX_FMT_BGRA;
    hwframeCtx->sw_format = videoPixelFormat;
    hwframeCtx->width = renderTarget->width;
    hwframeCtx->height = renderTarget->height;
    hwframeCtx->initial_pool_size = 0;

    /* init ffmpeg cuda frame context */
    auto cuda_frame_ref = av_hwframe_ctx_alloc(cudaDeviceCtx);
    if (!cuda_frame_ref) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't allocate CUDA frame context\n");
        throw std::runtime_error("Error: Couldn't allocate CUDA frame context");
    }
    auto cudaHwframeCtx = reinterpret_cast<AVHWFramesContext*>(cuda_frame_ref->data);
    cudaHwframeCtx->format = AV_PIX_FMT_CUDA;
    // cudaHwframeCtx->sw_format = AV_PIX_FMT_BGRA;
    cudaHwframeCtx->sw_format = videoPixelFormat;
    // cudaHwframeCtx->sw_format = AV_PIX_FMT_YUV420P; //Try YUV420P
    cudaHwframeCtx->width = renderTarget->width;
    cudaHwframeCtx->height = renderTarget->height;
    cudaHwframeCtx->initial_pool_size = 0;

    ret = av_hwframe_ctx_init(cuda_frame_ref);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't initialize CUDA frame context: %s\n", av_err2str(ret));
        throw std::runtime_error("Error: Couldn't initialize CUDA frame context");
    }

    this->cudaFrameCtx = cuda_frame_ref;
#endif

    /* Setup codec to encode output (video to URL) */
#ifdef __APPLE__
    std::string encoderName = "h264_videotoolbox";
#elif __linux__
    std::string encoderName = "h264_nvenc";
#else
    std::string encoderName = "libx264";
#endif
    auto outputCodec = avcodec_find_encoder_by_name(encoderName.c_str());
    std::cout << "Encoder: " << encoderName << std::endl;
    if (!outputCodec) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't allocate encoder.\n");
        throw std::runtime_error("Video Streamer could not be created.");
    }

    codecCtx = avcodec_alloc_context3(outputCodec);
    if (!codecCtx) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't allocate codec context.\n");
        throw std::runtime_error("Video Streamer could not be created.");
    }

#ifndef __APPLE__
    codecCtx->pix_fmt = AV_PIX_FMT_CUDA;
    codecCtx->hw_frames_ctx = av_buffer_ref(cudaFrameCtx);
#else
    codecCtx->pix_fmt = videoPixelFormat;
#endif
    codecCtx->width = width;
    codecCtx->height = height;
    codecCtx->time_base = {1, targetFrameRate};
    codecCtx->framerate = {targetFrameRate, 1};
    codecCtx->bit_rate = targetBitRate;

    // Set zero latency
    codecCtx->max_b_frames = 0;
    codecCtx->gop_size = 0;
    av_opt_set_int(codecCtx->priv_data, "zerolatency", 1, 0);
    av_opt_set_int(codecCtx->priv_data, "delay", 0, 0);

    ret = avcodec_open2(codecCtx, outputCodec, nullptr);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't open codec: %s\n", av_err2str(ret));
        throw std::runtime_error("Video Streamer could not be created.");
    }

    /* Setup output (to write video to URL) */
    ret = avformat_alloc_output_context2(&outputFormatCtx, nullptr, "mpegts", videoURL.c_str());
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Could not allocate output context: %s\n", av_err2str(ret));
        throw std::runtime_error("Video Streamer could not be created.");
    }

    outputFormatCtx->interrupt_callback.callback = interrupt_callback;
    outputFormatCtx->interrupt_callback.opaque = &shouldTerminate;

    outputVideoStream = avformat_new_stream(outputFormatCtx, outputCodec);
    if (!outputVideoStream) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Could not create new video stream.\n");
        throw std::runtime_error("Video Streamer could not be created.");
    }

    // Open output URL
    ret = avio_open(&outputFormatCtx->pb, this->videoURL.c_str(), AVIO_FLAG_WRITE);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Cannot open output URL\n");
        throw std::runtime_error("Video Streamer could not be created.");
    }

    ret = avformat_write_header(outputFormatCtx, nullptr);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error writing header\n");
        throw std::runtime_error("Video Streamer could not be created.");
    }

    swsCtx = sws_getContext(width, height, openglPixelFormat,
                                   width, height, videoPixelFormat,
                                   SWS_BICUBIC, nullptr, nullptr, nullptr);
    if (!swsCtx) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Could not allocate conversion context\n");
        throw std::runtime_error("Video Streamer could not be created.");
    }

    rgbData = new uint8_t[width * height * 3];

    /* setup frame */
#ifndef __APPLE__
    frame->format = AV_PIX_FMT_CUDA;
    frame->width = width;
    frame->height = height;
    frame->hw_frames_ctx = av_buffer_ref(cudaFrameCtx);

    ret = av_hwframe_get_buffer(cudaFrameCtx, frame, 0);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't allocate frame data: %s\n", av_err2str(ret));
        throw std::runtime_error("Error: Couldn't allocate frame data");
    }
#else
    frame->format = videoPixelFormat;
    frame->width = width;
    frame->height = height;

    ret = av_frame_get_buffer(frame, 0);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Could not allocate frame data: %s\n", av_err2str(ret));
        return;
    }

    ret = av_frame_make_writable(frame);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Could not make frame writable: %s\n", av_err2str(ret));
        return;
    }
#endif

    videoStreamerThread = std::thread(&VideoStreamer::encodeAndSendFrames, this);
}

#ifndef __APPLE__
CUdevice VideoStreamer::findCudaDevice() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: No CUDA devices found\n");
        return -1;
    }

    CUdevice device;

    char name[100];
    cuDeviceGet(&device, 0);
    cuDeviceGetName(name, 100, device);
    av_log(nullptr, AV_LOG_INFO, "CUDA Device: %s\n", name);

    return device;
}

int VideoStreamer::initCuda() {
    CUdevice device = findCudaDevice();
    if (device == -1) {
        return -1;
    }

    cudaError_t cudaErr = cudaGraphicsGLRegisterImage(&cudaResource, renderTarget->colorBuffer.ID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly);
    if (cudaErr != cudaSuccess) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't register CUDA image: %s\n", cudaGetErrorString(cudaErr));
        return -1;
    }

    return 0;
}
#endif

void VideoStreamer::sendFrame(unsigned int poseID) {
    /* Copy frame from OpenGL texture to AVFrame */
    uint64_t startCopyTime = av_gettime();

#ifndef __APPLE__
    cudaError_t cudaErr;

    cudaErr = cudaGraphicsMapResources(1, &cudaResource);
    if (cudaErr != cudaSuccess) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't map CUDA resources: %s\n", cudaGetErrorString(cudaErr));
        return;
    }

    cudaErr = cudaGraphicsSubResourceGetMappedArray(&cudaBuffer, cudaResource, 0, 0);
    if (cudaErr != cudaSuccess) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't get CUDA buffer: %s\n", cudaGetErrorString(cudaErr));
        return;
    }
#endif

    // lock frame mutex
    std::lock_guard<std::mutex> lock(frameMutex);

#ifndef __APPLE__
    cudaErr = cudaMemcpy2DFromArray(frame->data[0], frame->linesize[0],
                                    cudaBuffer,
                                    0, 0, width * 4, height,
                                    cudaMemcpyDeviceToHost);
    if (cudaErr != cudaSuccess) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't copy CUDA buffer: %s\n", cudaGetErrorString(cudaErr));
        return;
    }
#else
    renderTarget->bind();
    glReadPixels(0, 0, renderTarget->width, renderTarget->height, GL_RGB, GL_UNSIGNED_BYTE, rgbData);
    renderTarget->unbind();

    const uint8_t* srcData[] = { rgbData };
    int srcStride[] = { static_cast<int>(renderTarget->width * 3) }; // RGB has 3 bytes per pixel

    sws_scale(swsCtx, srcData, srcStride, 0, renderTarget->height, frame->data, frame->linesize);
#endif

    this->poseID = poseID;

    // tell thread to send frame
    frameReady = true;
    cv.notify_one();

#ifndef __APPLE__
    cudaErr = cudaGraphicsUnmapResources(1, &cudaResource);
    if (cudaErr != cudaSuccess) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't unmap CUDA resources: %s\n", cudaGetErrorString(cudaErr));
        return;
    }
#endif

    stats.timeToCopyFrame = (av_gettime() - startCopyTime) / MICROSECONDS_IN_MILLISECOND;
}

void VideoStreamer::encodeAndSendFrames() {
    sendFrames = true;

    uint64_t prevTime = av_gettime();

    int ret;
    while (true) {
        // wait for frame to be ready
        std::unique_lock<std::mutex> lock(frameMutex);
        cv.wait(lock, [&] { return frameReady; });

        if (sendFrames) {
            frameReady = false;
        }
        else {
            break;
        }

        /* Encode frame */
        {
            uint64_t startEncodeTime = av_gettime();

            // send frame to encoder
            ret = avcodec_send_frame(codecCtx, frame);
            if (ret < 0 || ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                av_log(nullptr, AV_LOG_ERROR, "Error: Could not send frame to output encoder: %s\n", av_err2str(ret));
                continue;
            }

            // get packet from encoder
            ret = avcodec_receive_packet(codecCtx, packet);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                continue;
            }
            else if (ret < 0) {
                av_log(nullptr, AV_LOG_ERROR, "Error: Could not receive frame from encoder: %s\n", av_err2str(ret));
                continue;
            }

            packet->pts = poseID; // framesSent * (outputFormatCtx->streams[0]->time_base.den) / targetFrameRate;
            packet->dts = packet->pts;

            stats.timeToEncode = (av_gettime() - startEncodeTime) / MICROSECONDS_IN_MILLISECOND;
        }

        /* Send frame to output URL */
        {
            uint64_t startWriteTime = av_gettime();

            // send packet to output URL
            ret = av_interleaved_write_frame(outputFormatCtx, packet);
            if (ret < 0) {
                av_packet_unref(packet);
                av_log(nullptr, AV_LOG_ERROR, "Error writing frame\n");
                continue;
            }

            av_packet_unref(packet);

            framesSent++;

            stats.timeToSendFrame = (av_gettime() - startWriteTime) / MICROSECONDS_IN_MILLISECOND;
        }

        uint64_t elapsedTime = (av_gettime() - prevTime);
        if (elapsedTime < (1.0f / targetFrameRate * MICROSECONDS_IN_MILLISECOND)) {
            av_usleep((1.0f / targetFrameRate * MICROSECONDS_IN_MILLISECOND) - elapsedTime);
        }
        stats.totalTimeToSendFrame = (av_gettime() - prevTime) / MICROSECONDS_IN_MILLISECOND;

        prevTime = av_gettime();
    }
}

void VideoStreamer::cleanup() {
    shouldTerminate = true;
    sendFrames = false;

    // send dummy frame to unblock thread
    frameReady = true;
    cv.notify_one();

    if (videoStreamerThread.joinable()) {
        videoStreamerThread.join();
    }

    avio_closep(&outputFormatCtx->pb);
    avformat_close_input(&outputFormatCtx);
    avformat_free_context(outputFormatCtx);

#ifndef __APPLE__
    cudaDeviceSynchronize();
    cudaError_t cudaErr = cudaGraphicsUnregisterResource(cudaResource);
    if (cudaErr != cudaSuccess) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't unregister CUDA resource: %s\n", cudaGetErrorString(cudaErr));
    }
#endif

    av_frame_free(&frame);
    av_packet_unref(packet);
    av_packet_free(&packet);

    delete[] rgbData;
}
