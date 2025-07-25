#include <cstring>
#include <spdlog/spdlog.h>
#include <Utils/TimeUtils.h>
#include <VideoStreamer.h>

using namespace quasar;

VideoStreamer::VideoStreamer(
        const RenderTargetCreateParams& params,
        const std::string& videoURL,
        int targetFrameRate,
        int targetBitRateMbps,
        const std::string& formatName)
    : targetFrameRate(targetFrameRate)
    , targetBitRate(targetBitRateMbps * BYTES_PER_MEGABYTE)
    , formatName(formatName)
    , videoURL(videoURL)
    , videoWidth(params.width + poseIDOffset)
    , videoHeight(params.height)
    , RenderTarget(params)
#if !defined(__APPLE__) && !defined(__ANDROID__)
    , cudaGLImage(colorBuffer)
#endif
{
    renderTargetCopy = new RenderTarget({
        .width = width,
        .height = height,
        .internalFormat = colorBuffer.internalFormat,
        .format = colorBuffer.format,
        .type = colorBuffer.type,
        .wrapS = colorBuffer.wrapS,
        .wrapT = colorBuffer.wrapT,
        .minFilter = colorBuffer.minFilter,
        .magFilter = colorBuffer.magFilter,
        .multiSampled = colorBuffer.multiSampled,
    });

    rgbaVideoFrameData.resize(videoWidth * videoHeight * 4);
#if defined(__APPLE__) || defined(__ANDROID__)
    openglFrameData.resize(width * height * 4);
#endif

    gst_init(nullptr, nullptr);

    // Parse host and port
    auto colonPos = videoURL.find(':');
    std::string host = videoURL.substr(0, colonPos);
    int port = std::stoi(videoURL.substr(colonPos + 1));

    std::string pipelineStr =
        "appsrc name=oglsrc0 is-live=true format=time "
        "caps=video/x-raw,format=RGBA,width=" + std::to_string(videoWidth) +
        ",height=" + std::to_string(videoHeight) +
        ",framerate=" + std::to_string(targetFrameRate) + "/1 ! "
        "queue ! videoconvert ! video/x-raw,format=I420 ! "
        "x264enc tune=zerolatency bitrate=" + std::to_string(targetBitRate / 1024) +
        " speed-preset=ultrafast ! rtph264pay config-interval=1 pt=96 name=pay0 ! "
        "udpsink host=" + host + " port=" + std::to_string(port);

    GError* error = nullptr;
    pipeline = gst_parse_launch(pipelineStr.c_str(), &error);
    if (!pipeline || error) {
        spdlog::error("GStreamer pipeline error: {}", error ? error->message : "unknown");
        g_error_free(error);
        throw std::runtime_error("Failed to create GStreamer pipeline.");
    }

    appsrc = gst_bin_get_by_name(GST_BIN(pipeline), "oglsrc0");
    g_object_set(G_OBJECT(appsrc),
                 "is-live", TRUE,
                 "format", GST_FORMAT_TIME,
                 "do-timestamp", TRUE,
                 nullptr);

    // Add pad probe to rtph264pay (pay0)
    GstElement* payloader = gst_bin_get_by_name(GST_BIN(pipeline), "pay0");
    GstPad* srcPad = gst_element_get_static_pad(payloader, "src");
    gst_pad_add_probe(srcPad, GST_PAD_PROBE_TYPE_BUFFER,
        [](GstPad*, GstPadProbeInfo* info, gpointer userData) -> GstPadProbeReturn {
            auto* totalBytesSent = static_cast<std::atomic<uint64_t>*>(userData);
            if (GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_BUFFER) {
                GstBuffer* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
                if (buffer) {
                    totalBytesSent->fetch_add(gst_buffer_get_size(buffer));
                }
            }
            return GST_PAD_PROBE_OK;
        },
        &this->totalBytesSent, nullptr);

    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    videoStreamerThread = std::thread(&VideoStreamer::encodeAndSendFrames, this);
    spdlog::info("Created VideoStreamer (GStreamer) that sends to: {}", videoURL);
}

VideoStreamer::~VideoStreamer() {
    shouldTerminate = true;
    videoReady = false;

    frameReady = true;
    cv.notify_one();

    if (videoStreamerThread.joinable())
        videoStreamerThread.join();

    if (appsrc) {
        gst_app_src_end_of_stream(GST_APP_SRC(appsrc));
        gst_object_unref(appsrc);
    }

    if (pipeline) {
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(pipeline);
    }

    delete renderTargetCopy;
}

float VideoStreamer::getFrameRate() {
    return 1.0f / timeutils::millisToSeconds(stats.totalTimeToSendMs);
}

void VideoStreamer::setTargetFrameRate(int targetFrameRate) {
    this->targetFrameRate = targetFrameRate;
}

void VideoStreamer::setTargetBitRate(uint targetBitRate) {
    this->targetBitRate = targetBitRate;
}

void VideoStreamer::sendFrame(pose_id_t poseID) {
#if !defined(__APPLE__) && !defined(__ANDROID__)
    bind();
    blitToRenderTarget(*renderTargetCopy);
    unbind();

    cudaArray* cudaBuffer = cudaGLImage.getArray();
    {
        std::lock_guard<std::mutex> lock(m);
        cudaBufferQueue.push({ poseID, cudaBuffer });
#else
    {
        std::lock_guard<std::mutex> lock(m);
        this->poseID = poseID;

        bind();
        glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, openglFrameData.data());
        unbind();
#endif
        frameReady = true;
    }
    cv.notify_one();
}

void VideoStreamer::packPoseIDIntoVideoFrame(pose_id_t poseID) {
    for (int i = 0; i < poseIDOffset; ++i) {
        uint8_t value = (poseID & (1 << i)) ? 255 : 0;
        for (int j = 0; j < videoHeight; ++j) {
            int index = j * videoWidth * 4 + (videoWidth - 1 - i) * 4;
            rgbaVideoFrameData[index + 0] = value;
            rgbaVideoFrameData[index + 1] = value;
            rgbaVideoFrameData[index + 2] = value;
            rgbaVideoFrameData[index + 3] = 255;
        }
    }
}

void VideoStreamer::encodeAndSendFrames() {
    videoReady = true;

    time_t prevTime = timeutils::getTimeMicros();
    time_t lastBitrateCalcTime = prevTime;

    while (true) {
        double frameIntervalSec = 1.0 / targetFrameRate;
        time_t frameStart = timeutils::getTimeMicros();

        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock, [this]() { return frameReady || shouldTerminate; });

        if (shouldTerminate) break;
        frameReady = false;

        time_t startCopyTime = timeutils::getTimeMicros();
#if !defined(__APPLE__) && !defined(__ANDROID__)
        auto cudaStruct = cudaBufferQueue.front();
        pose_id_t poseIDToSend = cudaStruct.poseID;
        cudaArray* cudaBuffer = cudaStruct.buffer;
        cudaBufferQueue.pop();

        lock.unlock();

        CHECK_CUDA_ERROR(cudaMemcpy2DFromArray(rgbaVideoFrameData.data(),
                                               videoWidth * 4,
                                               cudaBuffer,
                                               0, 0,
                                               width * 4, height,
                                               cudaMemcpyDeviceToHost));
#else
        pose_id_t poseIDToSend = this->poseID;
        for (int row = 0; row < height; ++row) {
            std::memcpy(&rgbaVideoFrameData[row * videoWidth * 4],
                        &openglFrameData[row * width * 4],
                        width * 4);
        }
        lock.unlock();
#endif
        // Pack the pose ID into the video frame
        packPoseIDIntoVideoFrame(poseIDToSend);
        stats.timeToCopyFrameMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startCopyTime);

        // Push to GStreamer
        time_t startEncode = timeutils::getTimeMicros();

        GstBuffer* buffer = gst_buffer_new_allocate(nullptr, rgbaVideoFrameData.size(), nullptr);
        GstMapInfo map;
        gst_buffer_map(buffer, &map, GST_MAP_WRITE);
        std::memcpy(map.data, rgbaVideoFrameData.data(), rgbaVideoFrameData.size());
        gst_buffer_unmap(buffer, &map);

        GstClockTime pts = gst_util_uint64_scale(framesSent, GST_SECOND, targetFrameRate);
        GstClockTime duration = gst_util_uint64_scale(1, GST_SECOND, targetFrameRate);

        GST_BUFFER_PTS(buffer) = pts;
        GST_BUFFER_DTS(buffer) = pts;
        GST_BUFFER_DURATION(buffer) = duration;

        GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(appsrc), buffer);
        if (ret != GST_FLOW_OK) {
            spdlog::error("Failed to push buffer to GStreamer: {}", static_cast<int>(ret));
        }
        framesSent++;

        time_t frameEnd = timeutils::getTimeMicros();

        stats.timeToEncodeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startEncode);

        // Bitrate calculation from pad probe
        time_t now = timeutils::getTimeMicros();
        time_t deltaSec = timeutils::microsToSeconds(now - lastBitrateCalcTime);
        if (deltaSec > 0.1) {
            stats.bitrateMbps = ((totalBytesSent * 8.0) / BYTES_PER_MEGABYTE) / deltaSec;
            totalBytesSent = 0;
            lastBitrateCalcTime = now;
        }

        stats.timeToSendMs = timeutils::microsToMillis(frameEnd - frameStart);

        // sleep to maintain target frame rate
        double elapsedTimeSec = timeutils::microsToSeconds(frameEnd - frameStart);
        if (elapsedTimeSec < frameIntervalSec) {
            std::this_thread::sleep_for(std::chrono::microseconds(
                (int)(timeutils::secondsToMicros(frameIntervalSec - elapsedTimeSec))));
        }

        stats.totalTimeToSendMs = timeutils::microsToMillis(timeutils::getTimeMicros() - prevTime);
        prevTime = timeutils::getTimeMicros();
    }
}
