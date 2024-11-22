#include <Recorder.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <filesystem> 
#include "Utils/FileIO.h"

Recorder::~Recorder() {
    if (running) {
        stop();
    }
}

void Recorder::start() {
    running = true;
    recordingStartTime = std::chrono::steady_clock::now();
    lastCaptureTime = std::chrono::steady_clock::now();
    for (int i = 0; i < NUM_SAVE_THREADS; ++i) {
        saveThreadPool.emplace_back(&Recorder::saveFrames, this);
    }
    if (outputFormat == OutputFormat::MP4) {
        initializeFFmpeg();
    }
}

void Recorder::stop() {
    if (!running) {
        return;
    }
    
    running = false;
    queueCV.notify_all();
    for (auto& thread : saveThreadPool) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    saveThreadPool.clear();

    if (outputFormat == OutputFormat::MP4) {
        if (codecContext) {
            int ret = avcodec_send_frame(codecContext, nullptr);
            while (ret >= 0) {
                AVPacket* pkt = av_packet_alloc();
                ret = avcodec_receive_packet(codecContext, pkt);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    av_packet_free(&pkt);
                    break;
                }
                if (ret < 0) {
                    av_packet_free(&pkt);
                    break;
                }
                
                pkt->stream_index = videoStream->index;
                av_packet_rescale_ts(pkt, codecContext->time_base, videoStream->time_base);
                av_interleaved_write_frame(formatContext, pkt);
                av_packet_free(&pkt);
            }
        }
        
        finalizeFFmpeg();
    }

    frameQueue = std::queue<FrameData>();
}

void Recorder::captureFrame(GeometryBuffer& gbuffer, Camera& camera) {
    auto currentTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - recordingStartTime).count();
    if (currentTime - lastCaptureTime >= frameInterval) {
        std::vector<unsigned char> frameData(captureTarget->width * captureTarget->height * 4);
        
        gbuffer.colorBuffer.bind(0);
        glReadPixels(0, 0, captureTarget->width, captureTarget->height, GL_RGBA, GL_UNSIGNED_BYTE, frameData.data());
        gbuffer.colorBuffer.unbind();

        {
            std::lock_guard<std::mutex> lock(queueMutex);
            frameQueue.push(FrameData{std::move(frameData), camera.getPosition(), camera.getRotationEuler(), elapsedTime});
        }
        queueCV.notify_one();

        lastCaptureTime = currentTime;
    }
}

void Recorder::saveFrames() {
    if (outputFormat != OutputFormat::MP4) {
        if (!std::filesystem::exists(outputPath)) {
            std::filesystem::create_directories(outputPath);
        }
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

        if (outputFormat == OutputFormat::MP4) {   
            AVFrame* inputFrame = av_frame_alloc();
            inputFrame->format = AV_PIX_FMT_RGBA;
            inputFrame->width = captureTarget->width;
            inputFrame->height = captureTarget->height;
            
            std::vector<unsigned char> flippedData(frameData.frame.size());
            const int stride = captureTarget->width * 4;
            for (int y = 0; y < captureTarget->height; ++y) {
                std::memcpy(
                    flippedData.data() + y * stride,
                    frameData.frame.data() + (captureTarget->height - 1 - y) * stride,
                    stride
                );
            }
            
            av_image_fill_arrays(inputFrame->data, inputFrame->linesize, flippedData.data(), 
                AV_PIX_FMT_RGBA, inputFrame->width, inputFrame->height, 1);

            sws_scale(swsContext, inputFrame->data, inputFrame->linesize, 0, inputFrame->height, frame->data, frame->linesize);
            
            frame->pts = frameData.pts;

            int ret = avcodec_send_frame(codecContext, frame);
            while (ret >= 0) {
                AVPacket* pkt = av_packet_alloc();
                pkt->data = nullptr;
                pkt->size = 0;

                ret = avcodec_receive_packet(codecContext, pkt);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    break;
                } else if (ret < 0) {
                    std::cerr << "Error encoding frame: " << ret << std::endl;
                    break;
                }

                pkt->stream_index = videoStream->index;
                av_packet_rescale_ts(pkt, codecContext->time_base, videoStream->time_base);
                av_interleaved_write_frame(formatContext, pkt);
                av_packet_unref(pkt);
            }
            av_frame_free(&inputFrame);
            frameIndex++;
        } else {
            size_t currentFrame = frameCount++;
            std::stringstream ss;
            ss << outputPath << "/frame_" << std::setw(6) << std::setfill('0') << currentFrame << "." << (outputFormat == OutputFormat::PNG ? "png" : "jpg");
            std::string filename = ss.str();

            try {
                FileIO::flipVerticallyOnWrite(true);
                if (outputFormat == OutputFormat::PNG) {
                    FileIO::saveAsPNG(filename, captureTarget->width, captureTarget->height, 4, frameData.frame.data());
                } else {
                    FileIO::saveAsJPG(filename, captureTarget->width, captureTarget->height, 4, frameData.frame.data());
                }

                {
                    std::lock_guard<std::mutex> lock(queueMutex);
                    std::ofstream pathFile(outputPath + "/camera_path.txt", std::ios::app);
                    pathFile << std::fixed << std::setprecision(4)
                             << frameData.position.x << " "
                             << frameData.position.y << " "
                             << frameData.position.z << " "
                             << frameData.euler.x << " "
                             << frameData.euler.y << " "
                             << frameData.euler.z << std::endl;
                    pathFile.close();
                }

                std::cout << "Saved frame: " << filename << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Error saving frame: " << filename << " - " << e.what() << std::endl;
            }
        }
    }
}

void Recorder::setOutputPath(const std::string& path) {
    outputPath = path;
}

void Recorder::setFrameRate(int fps) {
    frameInterval = std::chrono::duration<double>(1.0 / fps);
    lastCaptureTime = std::chrono::steady_clock::now();
    frameIndex = 0;
}

void Recorder::initializeFFmpeg() {
    avformat_alloc_output_context2(&formatContext, nullptr, nullptr, (outputPath + "/output.mp4").c_str());

    const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    codecContext = avcodec_alloc_context3(codec);
    codecContext->bit_rate = 5000000;
    codecContext->width = captureTarget->width;
    codecContext->height = captureTarget->height;
    
    int fps = static_cast<int>(1.0 / frameInterval.count());
    codecContext->time_base = (AVRational){1, 1000};
    codecContext->framerate = (AVRational){fps, 1};
    
    codecContext->gop_size = 12;
    codecContext->max_b_frames = 1;
    codecContext->pix_fmt = AV_PIX_FMT_YUV420P;

    AVDictionary* opt = nullptr;
    av_dict_set(&opt, "preset", "medium", 0);
    av_dict_set(&opt, "crf", "23", 0);
    av_dict_set(&opt, "threads", "auto", 0);
    av_dict_set(&opt, "refs", "3", 0);
    av_dict_set(&opt, "me_range", "16", 0);

    if (formatContext->oformat->flags & AVFMT_GLOBALHEADER) {
        codecContext->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    int ret = avcodec_open2(codecContext, codec, &opt);
    av_dict_free(&opt);
    if (ret < 0) {
        throw std::runtime_error("Failed to open codec");
    }

    videoStream = avformat_new_stream(formatContext, nullptr);
    videoStream->time_base = codecContext->time_base;
    avcodec_parameters_from_context(videoStream->codecpar, codecContext);

    if (!(formatContext->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&formatContext->pb, (outputPath + "/output.mp4").c_str(), AVIO_FLAG_WRITE);
        if (ret < 0) {
            throw std::runtime_error("Failed to open output file");
        }
    }

    ret = avformat_write_header(formatContext, nullptr);
    if (ret < 0) {
        throw std::runtime_error("Failed to write header");
    }

    swsContext = sws_getContext(
        captureTarget->width, captureTarget->height, AV_PIX_FMT_RGBA,
        captureTarget->width, captureTarget->height, AV_PIX_FMT_YUV420P,
        SWS_BILINEAR, nullptr, nullptr, nullptr);

    frame = av_frame_alloc();
    frame->format = AV_PIX_FMT_YUV420P;
    frame->width = codecContext->width;
    frame->height = codecContext->height;
    av_frame_get_buffer(frame, 0);
}

void Recorder::finalizeFFmpeg() {
    if (formatContext) {
        if (formatContext->pb != nullptr) {
            av_write_trailer(formatContext);
            if (!(formatContext->oformat->flags & AVFMT_NOFILE)) {
                avio_closep(&formatContext->pb);
            }
        }
    }
    
    if (swsContext) {
        sws_freeContext(swsContext);
        swsContext = nullptr;
    }

    if (frame) {
        av_frame_free(&frame);
        frame = nullptr;
    }

    if (codecContext) {
        avcodec_free_context(&codecContext);
        codecContext = nullptr;
    }
    
    if (formatContext) {
        avformat_free_context(formatContext);
        formatContext = nullptr;
    }
    
    frameIndex = 0;
}

void Recorder::setOutputFormat(OutputFormat format) {
    if (running) {
        return;
    }

    outputFormat = format;
}
