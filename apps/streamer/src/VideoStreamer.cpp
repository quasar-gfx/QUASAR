#include <VideoStreamer.h>
#ifndef __APPLE__
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#undef av_err2str
#define av_err2str(errnum) av_make_error_string((char*)__builtin_alloca(AV_ERROR_MAX_STRING_SIZE), AV_ERROR_MAX_STRING_SIZE, errnum)

int VideoStreamer::initializeCudaContext(std::string& gpuName, int width, int height, GLuint texture) {
    std::cout << "initializing Cuda Context" << std::endl;
    if (getDeviceName(gpuName) < 0) {
        std::cerr << "Failed to get CUDA device name." << std::endl;
        return -1;
    }
    std::cout << "Graphic Device Name:" << gpuName << std::endl;

    // Create CUDA Hardware Context for ffmpeg
    AVBufferRef* m_avBufferRefDevice;
    int ret = av_hwdevice_ctx_create(&m_avBufferRefDevice, AV_HWDEVICE_TYPE_CUDA, gpuName.c_str(), NULL, 0);
    if (ret < 0) {
        std::cerr << "Failed to create a hardware device context." << std::endl;
        return false;
    }
    std::cout << "HW Device Context created" << std::endl;
    
    // Retrieve and cast the hardware device context to a CUDA device context for accessing CUDA-specific functionalities.
    
    AVHWDeviceContext* hwDevContext = reinterpret_cast<AVHWDeviceContext*>(m_avBufferRefDevice->data);
    std:: cout << hwDevContext << std::endl;
    AVCUDADeviceContext* cudaDevCtx = reinterpret_cast<AVCUDADeviceContext*>(hwDevContext->hwctx);
    std:: cout << cudaDevCtx << std::endl;
    CUresult res;
    m_cuContext = &(cudaDevCtx->cuda_ctx);
    std:: cout << m_cuContext << std::endl;
    unsigned int *version = nullptr;
    version = (unsigned int*)malloc(sizeof(unsigned int));
    res = cuCtxGetApiVersion(*m_cuContext, version);
    std::cout << "cuRes: " << res << " Context Api Version: " << *version << std::endl;
    
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
    std::cout << "Hw Frame Context " << std::endl;
    
    
    CUcontext oldCtx;
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
        printf("%s:%d %s failed. \n code = %s, message = %s\n", file, line, op, err_name, err_message);
        return false;
    }
    return true;
}

int VideoStreamer::getDeviceName(std::string& gpuName) {
    // Setup the cuda context for hardware encoding with ffmpeg
    int iGpu = 0;
    CUresult res;
    res = cuInit(0);
    std::cout << res << std::endl;
    int nGpu = 0;
    cuDeviceGetCount(&nGpu);
    if (iGpu < 0 || iGpu >= nGpu) {
        std::cout << "GPU ordinal out of range. Should be within [" << 0 << ", " << nGpu - 1 << "]" << std::endl;
        return 1;
    }

    CUdevice cuDevice = 0;
    cuDeviceGet(&cuDevice, iGpu);
    char szDeviceName[80];
    cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice);
    gpuName = szDeviceName;

    printf("Device %d name is %s\n", cuDevice, szDeviceName);
    return 0;
}

int VideoStreamer::start(Texture* texture, const std::string outputUrl) {
    this->sourceTexture = texture;
    this->outputUrl = outputUrl;

#ifndef __APPLE__
    std::string gpuName;
    if (getDeviceName(gpuName) != 0) {
        return 1;
    }
    std::cout << "GPU in use: " << gpuName << std::endl;
#endif

    /* BEGIN: Setup codec to encode output (video to URL) */
#ifdef __APPLE__
    const AVCodec *outputCodec = avcodec_find_encoder_by_name("h264_videotoolbox");
    std::cout << "Encoder: h264_videotoolbox" << std::endl;
#else
    const AVCodec *outputCodec = avcodec_find_encoder_by_name("h264_nvenc");
    std::cout << "Encoder: h264_nvenc" << std::endl;
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

    outputCodecContext->width = texture->width;
    outputCodecContext->height = texture->height;
    outputCodecContext->time_base = {1, frameRate};
    outputCodecContext->framerate = {frameRate, 1};
    outputCodecContext->pix_fmt = this->pixelFormat;
    outputCodecContext->bit_rate = 100000 * 1000;

    // Set zero latency
    outputCodecContext->max_b_frames = 0;
    outputCodecContext->gop_size = 0;
    av_opt_set_int(outputCodecContext->priv_data, "zerolatency", 1, 0);
    av_opt_set_int(outputCodecContext->priv_data, "delay", 0, 0);

    int ret = avcodec_open2(outputCodecContext, outputCodec, nullptr);
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

    conversionContext = sws_getContext(texture->width, texture->height, AV_PIX_FMT_RGBA,
                                                    texture->width, texture->height, this->pixelFormat,
                                                    SWS_BICUBIC, nullptr, nullptr, nullptr);
    if (!conversionContext) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Could not allocate conversion context\n");
        return -1;
    }

    rgbaData = new uint8_t[texture->width * texture->height * 4];
    int res = initializeCudaContext(gpuName, texture->width, texture->height, texture->ID);
    if (res < 0) {
        std::cout << "Initilization Cuda Context Failed" << std::endl;
    }
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
    std:: cout << m_cuContext << std::endl;
    std::cout << "Get context: " << cuRes << std::endl;
    
    unsigned int *version = nullptr;
    version = (unsigned int*)malloc(sizeof(unsigned int));
    cuRes = cuCtxGetApiVersion(*m_cuContext, version);
    std::cout << "cuRes: " << cuRes << " Context Api Version: " << *version << std::endl;

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
    std::cout << "cu Memcpy 2D: " << cuRes << std::endl;
    
    //release context
    cuRes = cuCtxPopCurrent(&oldCtx); // THIS IS ALLOWED TO FAIL
    std::cout << "Release Context: " << cuRes << std::endl;
    
    return 0;
}

void VideoStreamer::sendFrame() {
    static uint64_t lastTime = av_gettime();

    // get frame from decoder
    AVFrame *frame = av_frame_alloc();
    prepareEncode(frame);
    frame->format = this->pixelFormat;
    frame->width = sourceTexture->width;
    frame->height = sourceTexture->height;

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

    sourceTexture->bind(0);
    glReadPixels(0, 0, sourceTexture->width, sourceTexture->height, GL_RGBA, GL_UNSIGNED_BYTE, rgbaData);
    sourceTexture->unbind();

    const uint8_t* srcData[] = { rgbaData };
    int srcStride[] = { static_cast<int>(sourceTexture->width * 4) }; // RGBA has 4 bytes per pixel

    sws_scale(conversionContext, srcData, srcStride, 0, sourceTexture->height, frame->data, frame->linesize);

    // send packet to encoder
    ret = avcodec_send_frame(outputCodecContext, frame);
    if (ret < 0 || ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Could not send frame to output encoder: %s\n", av_err2str(ret));
        return;
    }

    // get packet from encoder
    ret = avcodec_receive_packet(outputCodecContext, &outputPacket);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
        av_frame_free(&frame);
        av_packet_unref(&outputPacket);
        return;
    }
    else if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error: Could not receive frame from encoder: %s\n", av_err2str(ret));
        return;
    }

    outputPacket.pts = framesSent * (outputFormatContext->streams[0]->time_base.den) / frameRate;
    outputPacket.dts = outputPacket.pts;

    av_usleep(1.0f / frameRate * 1000000.0f - (av_gettime() - lastTime));

    // send packet to output URL
    ret = av_interleaved_write_frame(outputFormatContext, &outputPacket);
    av_packet_unref(&outputPacket);
    av_frame_free(&frame);
    if (ret < 0) {
        av_log(nullptr, AV_LOG_ERROR, "Error writing frame\n");
        return;
    }

    framesSent++;

    lastTime = av_gettime();
}

void VideoStreamer::cleanup() {
    avio_closep(&outputFormatContext->pb);
    avformat_close_input(&outputFormatContext);
    avformat_free_context(outputFormatContext);
    sws_freeContext(conversionContext);
}
