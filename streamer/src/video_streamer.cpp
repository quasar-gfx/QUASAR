#include "video_streamer.hpp"

#undef av_err2str
#define av_err2str(errnum) av_make_error_string((char*)__builtin_alloca(AV_ERROR_MAX_STRING_SIZE), AV_ERROR_MAX_STRING_SIZE, errnum)

int VideoStreamer::initializeCudaContext(std::string& gpuName, int width, int height, GLuint texture) {
    std::cout << "initializing Cuda Context" << std::endl;
    if (getDeviceName(gpuName) < 0) {
        std::cerr << "Failed to get CUDA device name." << std::endl;
        return -1;
    }
    std::cout << "Graphic Device Name:" << gpuName << std::endl;
    cudaGLSetGLDevice(0);
    // Create CUDA Hardware Context for ffmpeg
    AVBufferRef* m_avBufferRefDevice;
    int ret = av_hwdevice_ctx_create(&m_avBufferRefDevice, AV_HWDEVICE_TYPE_CUDA, gpuName.c_str(), NULL, NULL);
    if (ret < 0) {
        std::cerr << "Failed to create a hardware device context." << std::endl;
        return false;
    }
    std::cout << "hw device ctx created" << std::endl;
    
    // Retrieve and cast the hardware device context to a CUDA device context for accessing CUDA-specific functionalities.
    AVHWDeviceContext* hwDevContext = reinterpret_cast<AVHWDeviceContext*>(m_avBufferRefDevice->data);
    AVCUDADeviceContext* cudaDevCtx = reinterpret_cast<AVCUDADeviceContext*>(hwDevContext->hwctx);
    m_cuContext = &(cudaDevCtx->cuda_ctx);

    
    AVBufferRef* m_avBufferRefFrame = av_hwframe_ctx_alloc(m_avBufferRefDevice);
    AVHWFramesContext* frameCtxPtr = reinterpret_cast<AVHWFramesContext*>(m_avBufferRefFrame->data);
    frameCtxPtr->width = width;
    frameCtxPtr->height = height;
    frameCtxPtr->sw_format = AV_PIX_FMT_0BGR32;
    frameCtxPtr->format = AV_PIX_FMT_CUDA;
    frameCtxPtr->device_ref = m_avBufferRefDevice;
    frameCtxPtr->device_ctx = hwDevContext;

    ret = av_hwframe_ctx_init(m_avBufferRefFrame);
    if (ret < 0) {
        std::cerr << "Failed to initialize a hardware frame context." << std::endl;
        return -1;
    }
    std::cout << "Hw Frame Context Created" << std::endl;
    
    CUresult res;
    CUcontext oldCtx;
    // cuInpTexRes = (CUgraphicsResource *)malloc(sizeof(CUgraphicsResource));e
    res = cuCtxPopCurrent(&oldCtx);
    std::cout << res << std::endl;
    res = cuCtxPushCurrent(*m_cuContext);
    std::cout << res << std::endl;
    res = cuGraphicsGLRegisterImage(&cuInpTexRes, texture, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY);
    std::cout << res << std::endl;
    //res = cuGraphicsGLRegisterBuffer(&cuInpTexRes, texture, CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY);
    res = cuCtxPopCurrent(&oldCtx);
    std::cout << res << std::endl;
    
    outputCodecContext->hw_device_ctx = av_buffer_ref(m_avBufferRefDevice);
    outputCodecContext->pix_fmt = AV_PIX_FMT_CUDA;
    outputCodecContext->hw_frames_ctx = av_buffer_ref(m_avBufferRefFrame);
    outputCodecContext->codec_type = AVMEDIA_TYPE_VIDEO;
    outputCodecContext->sw_pix_fmt = AV_PIX_FMT_0BGR32;

    m_memCpyStruct.srcXInBytes = 0;
    m_memCpyStruct.srcY = 0;
    m_memCpyStruct.srcMemoryType = CUmemorytype::CU_MEMORYTYPE_ARRAY;

    m_memCpyStruct.dstXInBytes = 0;
    m_memCpyStruct.dstY = 0;
    m_memCpyStruct.dstMemoryType = CUmemorytype::CU_MEMORYTYPE_DEVICE;
    return 0;
}

#define checkDriver(op) __check_cuda_driver((op), #op, __FILE__, __LINE__)
bool __check_cuda_driver(CUresult code, const char* op, const char* file, int line) {
    if(code != CUresult::CUDA_SUCCESS) {
        const char* err_name = nullptr;
        const char* err_message = nullptr;
        cuGetErrorName(code, &err_name);
        cuGetErrorString(code, &err_message);
        printf("%s:%d %d failed. \n code = %s, message = %s\n", file, line, op, err_name, err_message);
        return false;
    }
    return true;
}

int VideoStreamer::getDeviceName(std::string& gpuName) {
    //Setup the cuda context for hardware encoding with ffmpeg
    int iGpu = 0;
    CUresult res;
    checkDriver(cuInit(0));
    int nGpu = 0;
    cuDeviceGetCount(&nGpu);
    if (iGpu < 0 || iGpu >= nGpu)
    {
        std::cout << "GPU ordinal out of range. Should be within [" << 0 << ", " << nGpu - 1 << "]" << std::endl;
        return 1;
    }
    int driver_version = 0;
    if (!checkDriver(cuDriverGetVersion(&driver_version))) {
        return -1;
    }
    printf("Driver version is %d\n", driver_version);
    CUdevice cuDevice = 0;
    cuDeviceGet(&cuDevice, iGpu);
    char szDeviceName[80];
    cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice);
    gpuName = szDeviceName;
    printf("Device %d name is %s\n", cuDevice, szDeviceName);
    return 0;
}



int VideoStreamer::init(const std::string inputFileName, const std::string outputUrl) {
    this->inputFileName = inputFileName;
    this->outputUrl = outputUrl;

    /* BEGIN: Setup input (to read video from file) */
    int ret = avformat_open_input(&inputFormatContext, inputFileName.c_str(), nullptr, nullptr);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Cannot open input file: %s\n", av_err2str(ret));
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
        av_log(nullptr, AV_LOG_ERROR, "No video stream found in the input file.\n");
        return -1;
    }
    /* END: Setup input (to read video from file) */

    /* BEGIN: Setup codec to decode input (video from file) */
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
    /* END: Setup codec to decode input (video from file) */

    /* BEGIN: Setup codec to encode output (video to URL) */
#ifdef __APPLE__ // mac does not have nvenc
    const AVCodec *outputCodec = avcodec_find_encoder(AV_CODEC_ID_H264);
#else
    const AVCodec *outputCodec = avcodec_find_encoder_by_name("h264_nvenc");
#endif
    if (!outputCodec) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't allocate encoder.\n");
        return -1;
    }

    outputCodecContext = avcodec_alloc_context3(outputCodec);
    if (!outputCodecContext) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't allocate codec context.\n");
        return -1;
    }

    outputCodecContext->width = inputVideoStream->codecpar->width;
    outputCodecContext->height = inputVideoStream->codecpar->height;
    outputCodecContext->time_base = inputFormatContext->streams[videoStreamIndex]->time_base;
    outputCodecContext->framerate = inputFormatContext->streams[videoStreamIndex]->avg_frame_rate;
    outputCodecContext->pix_fmt = (AVPixelFormat)inputVideoStream->codecpar->format;
    outputCodecContext->bit_rate = 400000;

    // Set zero latency
    // outputCodecContext->max_b_frames = 0;
    // outputCodecContext->gop_size = 0;
    // av_opt_set_int(outputCodecContext->priv_data, "zerolatency", 1, 0);
    // av_opt_set_int(outputCodecContext->priv_data, "delay", 0, 0);

    ret = avcodec_open2(outputCodecContext, outputCodec, nullptr);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't open codec: %s\n", av_err2str(ret));
        return ret;
    }
    /* END: Setup codec to encode output (video to URL) */

    /* BEGIN: Setup output (to write video to URL) */
    // Open output URL
    ret = avformat_alloc_output_context2(&outputFormatContext, nullptr, "mpegts", outputUrl.c_str());
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Could not allocate output context: %s\n", av_err2str(ret));
        return ret;
    }

    outputVideoStream = avformat_new_stream(outputFormatContext, outputCodec);
    if (!outputVideoStream) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Could not create new video stream.\n");
        return -1;
    }

    ret = avcodec_parameters_from_context(outputVideoStream->codecpar, inputCodecContext);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Could not initialize output stream parameters: %s\n", av_err2str(ret));
        return ret;
    }

    // Open output URL
    ret = avio_open(&outputFormatContext->pb, outputUrl.c_str(), AVIO_FLAG_WRITE);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Cannot open output URL\n");
        return ret;
    }

    ret = avformat_write_header(outputFormatContext, nullptr);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error writing header\n");
        return ret;
    }
    /* END: Setup output (to write video to URL) */

    return 0;
}

int VideoStreamer::prepareEncode(AVFrame *frame) {
    std::cout << "Prepare Encode" << std::endl;
    
    //Perform cuda mem copy for input buffer
    CUresult cuRes;
    CUarray mappedArray;
    CUcontext oldCtx;

    //Get context
    cuRes = cuCtxPopCurrent(&oldCtx); // THIS IS ALLOWED TO FAIL
    cuRes = cuCtxPushCurrent(*m_cuContext);
    std::cout << "Get context: " << cuRes << std::endl;
    
    //Get Texture
    cuRes = cuGraphicsResourceSetMapFlags(cuInpTexRes, CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY);
    std::cout << cuRes << std::endl;
    cuRes = cuGraphicsMapResources(1, &cuInpTexRes, 0);
    std::cout << cuRes << std::endl;
    std::cout << "Get Texture: " << cuRes << std::endl;

    //Map texture to cuda array
    cuRes = cuGraphicsSubResourceGetMappedArray(&mappedArray, cuInpTexRes, 0, 0); // Nvidia says its good practice to remap each iteration as OGL can move things around
    std::cout << "Map Texture: " << cuRes << std::endl;
    
    //Release texture
    cuRes = cuCtxPopCurrent(&oldCtx); // THIS IS ALLOWED TO FAIL
    std::cout << "Release texture " << cuRes << std::endl;
    
    //Setup for memcopy
    m_memCpyStruct.srcArray = mappedArray;
    m_memCpyStruct.dstDevice = (CUdeviceptr)frame->data[0]; // Make sure to copy devptr as it could change, upon resize
    m_memCpyStruct.dstPitch = frame->linesize[0];   // Linesize is generated by hwframe_context
    m_memCpyStruct.WidthInBytes = frame->width * 4; //* 4 needed for each pixel
    m_memCpyStruct.Height = frame->height;          //Vanilla height for frame
    std::cout << "setup for memcopy" << std::endl;

    //Do memcpy
    cuRes = cuMemcpy2D(&m_memCpyStruct); 
    std::cout << "cu Memcpy 2D " << cuRes << std::endl;
    
    //release context
    cuRes = cuCtxPopCurrent(&oldCtx); // THIS IS ALLOWED TO FAIL
    std::cout << cuRes << std::endl;
    
    return 0;
}

int VideoStreamer::sendFrame() {
    std::cout << "-------------- Send Frame ------------------" << std::endl;
    static int64_t start_time = av_gettime();
    // read frame from file
    int ret = av_read_frame(inputFormatContext, &inputPacket);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error reading frame: %s\n", av_err2str(ret));
        return -1;
    }
    std::cout << "Frame read" << std::endl;

    if (inputPacket.stream_index == videoStreamIndex) {
        // send packet to decoder
        ret = avcodec_send_packet(inputCodecContext, &inputPacket);
        if (ret < 0) {
            av_log(nullptr, AV_LOG_ERROR, "Error: Could not send packet to input decoder: %s\n", av_err2str(ret));
            return ret;
        }
        std::cout << "Send Packet to input decoder" << std::endl;

        // get frame from decoder
        AVFrame *frame = av_frame_alloc();
        ret = avcodec_receive_frame(inputCodecContext, frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            av_frame_free(&frame);
            av_packet_unref(&inputPacket);
            return 1;
        }
        else if (ret < 0) {
            av_log(nullptr, AV_LOG_ERROR, "Error: Could not receive raw frame from input decoder: %s\n", av_err2str(ret));
            return ret;
        }
        std::cout << "Receive raw frame from input decoder" << std::endl;

        // send packet to encoder
        prepareEncode(frame);
        ret = avcodec_send_frame(outputCodecContext, frame);
        if (ret < 0 || ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            av_log(nullptr, AV_LOG_ERROR, "Error: Could not send frame to output encoder: %s\n", av_err2str(ret));
            return ret;
        }
        std::cout << "Send packet to encoder" << std::endl;

        // get packet from encoder
        ret = avcodec_receive_packet(outputCodecContext, &outputPacket);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            av_frame_free(&frame);
            av_packet_unref(&outputPacket);
            return 1;
        }
        else if (ret < 0) {
            av_log(nullptr, AV_LOG_ERROR, "Error: Could not receive frame from encoder: %s\n", av_err2str(ret));
            return ret;
        }

        outputPacket.stream_index = outputVideoStream->index;
        if (outputPacket.pts == AV_NOPTS_VALUE) {
            // write PTS
            AVRational time_base1 = inputVideoStream->time_base;
            // duration between 2 frames (us)
            int64_t calc_duration = (double)AV_TIME_BASE/av_q2d(inputVideoStream->r_frame_rate);
            // parameters
            outputPacket.pts = (double)(framesSent*calc_duration)/(double)(av_q2d(time_base1)*AV_TIME_BASE);
            outputPacket.dts = outputPacket.pts;
            outputPacket.duration = (double)calc_duration/(double)(av_q2d(time_base1)*AV_TIME_BASE);
        }

        // important to maintain FPS - delay
        AVRational time_base = inputVideoStream->time_base;
        AVRational time_base_q = {1, AV_TIME_BASE};
        int64_t pts_time = av_rescale_q(outputPacket.dts, time_base, time_base_q);
        int64_t now_time = av_gettime() - start_time;
        if (pts_time > now_time) {
            av_usleep(pts_time - now_time);
        }

        // convert PTS/DTS
        outputPacket.pts = av_rescale_q_rnd(inputPacket.pts, inputVideoStream->time_base, outputVideoStream->time_base,
                                        (AVRounding)(AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX));
        outputPacket.dts = av_rescale_q_rnd(inputPacket.dts, inputVideoStream->time_base, outputVideoStream->time_base,
                                        (AVRounding)(AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX));
        outputPacket.duration = av_rescale_q(inputPacket.duration, inputVideoStream->time_base, outputVideoStream->time_base);
        outputPacket.pos = -1;

        // send packet to output URL
        framesSent++;
        std::cout << "Sending frame " << framesSent << " to " << outputUrl << std::endl;
        ret = av_interleaved_write_frame(outputFormatContext, &outputPacket);
        av_packet_unref(&outputPacket);
        av_frame_free(&frame);
        if (ret < 0) {
            av_log(nullptr, AV_LOG_ERROR, "Error writing frame\n");
            return ret;
        }
    }

    return 0;
}

void VideoStreamer::cleanup() {
    avformat_close_input(&inputFormatContext);
    avformat_free_context(inputFormatContext);
    avio_closep(&outputFormatContext->pb);
    avformat_close_input(&outputFormatContext);
    avformat_free_context(outputFormatContext);
}
