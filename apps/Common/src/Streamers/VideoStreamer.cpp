#include <sstream>
#include <spdlog/spdlog.h>
#include <Streamers/VideoStreamer.h>
#include <Utils/TimeUtils.h>
#include <Networking/Utils.h>

using namespace quasar;

VideoStreamer::VideoStreamer(
        const RenderTargetCreateParams& params,
        const std::string& videoURL,
        uint maxFrameRate,
        uint targetBitRateMbps,
        bool useRTP)
    : videoURL(videoURL)
    , videoWidth(params.width + poseIDOffset)
    , videoHeight(params.height)
    , maxFrameRate(maxFrameRate)
    , targetBitRateKbps(targetBitRateMbps * 1000)
    , useRTP(useRTP)
    , RenderTarget(params)
#if defined(QUASAR_HAS_CUDA)
    , cudaGLImage(colorTexture)
#endif
{
    if (videoURL.empty()) {
        return;
    }

    gst_init(nullptr, nullptr);

    GstRegistry* registry = gst_registry_get();
    GList* factories = gst_registry_get_feature_list(registry, GST_TYPE_ELEMENT_FACTORY);
    std::ostringstream codecs;
    for (GList* l = factories; l != nullptr; l = l->next) {
        GstElementFactory* factory = GST_ELEMENT_FACTORY(l->data);
        const gchar* klass = gst_element_factory_get_metadata(factory, GST_ELEMENT_METADATA_KLASS);
        const gchar* name  = gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(factory));
        if (klass && g_strrstr(klass, "Encoder")) {
            codecs << name << " ";
        }
    }
    spdlog::debug("Available Encoders: {}", codecs.str());
    gst_plugin_feature_list_free(factories);

    auto [host, port] = networkutils::parseIPAddressAndPort(videoURL);

    std::string encoderParams;
    const uint gopFrames = std::max(1u, maxFrameRate); // 1 second GOP at worst case FPS
#if defined(QUASAR_HAS_CUDA)
    encoderParams = "nvh264enc preset=4 rc-mode=cbr zerolatency=true "
                    "bframes=0 gop-size=" + std::to_string(gopFrames);
#else
    encoderParams = "x264enc speed-preset=veryfast tune=zerolatency byte-stream=true "
                    "bframes=0 key-int-max=" + std::to_string(gopFrames);
#endif

    std::ostringstream oss;
    oss << "appsrc name=" << appSrcName << " is-live=true format=time "
        << "caps=video/x-raw,format=RGBA,width=" << videoWidth << ",height=" << videoHeight << " ! "
        << "queue leaky=upstream max-size-buffers=1 max-size-time=0 max-size-bytes=0 ! "
        << "videoconvert ! video/x-raw,format=" << format << " ! "
        << encoderParams << " bitrate=" << targetBitRateKbps << " ! "
        << "h264parse config-interval=1 name=" << h264ParseName << " ! ";
    if (useRTP) {
        oss << "rtph264pay config-interval=1 pt=96 name=" << payloaderName << " ! "
            << "udpsink host=" << host << " port=" << port << " sync=false";
    }
    else {
        oss << "srtsink uri=\"srt://" << host << ":" << port
            << "?mode=caller&latency=80\"";
    }
    std::string pipelineStr = oss.str();

    GError* error = nullptr;
    pipeline = gst_parse_launch(pipelineStr.c_str(), &error);
    if (!pipeline || error) {
        spdlog::error("GStreamer pipeline error: {}", error ? error->message : "unknown");
        g_error_free(error);
        throw std::runtime_error("Failed to create GStreamer pipeline.");
    }

    appsrc = gst_bin_get_by_name(GST_BIN(pipeline), appSrcName.c_str());
    g_object_set(G_OBJECT(appsrc),
                 "is-live", TRUE,
                 "format", GST_FORMAT_TIME,
                 "do-timestamp", TRUE,
                 nullptr);

    // Attach a buffer probe on the h264parse src pad to meter encoded bytes
    if (GstElement* h264parseEl = gst_bin_get_by_name(GST_BIN(pipeline), h264ParseName.c_str())) {
        if (GstPad* srcpad = gst_element_get_static_pad(h264parseEl, "src")) {
            gst_pad_add_probe(
                srcpad,
                GST_PAD_PROBE_TYPE_BUFFER,
                [](GstPad* /*pad*/, GstPadProbeInfo* info, gpointer user_data) -> GstPadProbeReturn {
                    auto* self = static_cast<VideoStreamer*>(user_data);
                    GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
                    if (buf && self) {
                        gsize sz = gst_buffer_get_size(buf);
                        self->encodedBytesTotal.fetch_add(static_cast<uint64_t>(sz), std::memory_order_relaxed);
                    }
                    return GST_PAD_PROBE_OK;
                },
                this,
                nullptr);
            gst_object_unref(srcpad);
        }
        gst_object_unref(h264parseEl);
    }
    else {
        spdlog::warn("Could not find '{}' element to probe bitrate.", h264ParseName);
    }

    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    videoStreamerThread = std::thread(&VideoStreamer::encodeAndSendFrames, this);
    spdlog::info("Created VideoStreamer (GStreamer) that sends to URL: {}://{}", useRTP ? "rtp" : "srt", videoURL);
}

VideoStreamer::~VideoStreamer() {
    if (videoURL.empty()) {
        return;
    }

    stop();
}

void VideoStreamer::stop() {
    shouldTerminate = true;

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
}

void VideoStreamer::sendFrame(pose_id_t poseID) {
    VideoFrame videoFrame;
    videoFrame.poseID = poseID;
    videoFrame.buffer.resize(width * height * 4);

#if defined(QUASAR_HAS_CUDA)
    cudaGLImage.copyArrayToHostAsync(
        width * 4,
        height,
        width * 4,
        videoFrame.buffer.data());
    cudaGLImage.synchronize();
#else
    readPixels(videoFrame.buffer.data());
#endif

    videoFrameQueue.enqueue(videoFrame);
}

void VideoStreamer::packPoseIDIntoVideoFrame(pose_id_t poseID, uint8_t* data) {
    for (int i = 0; i < poseIDOffset; i++) {
        uint8_t value = (poseID & (1 << i)) ? 255 : 0;
        for (int j = 0; j < videoHeight; j++) {
            int index = j * videoWidth * 4 + (videoWidth - 1 - i) * 4;
            data[index + 0] = value;
            data[index + 1] = value;
            data[index + 2] = value;
        }
    }
}

void VideoStreamer::encodeAndSendFrames() {
    double prevTime = timeutils::getTimeMicros();

    while (!shouldTerminate) {
        double frameIntervalSec = 1.0 / maxFrameRate;
        double frameStart = timeutils::getTimeMicros();

        double startTransferTimeMs = timeutils::getTimeMicros();
        VideoFrame videoFrame;
        if (!videoFrameQueue.try_dequeue(videoFrame)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        GstBuffer* buffer = gst_buffer_new_allocate(nullptr, videoWidth * videoHeight * 4, nullptr);
        GstMapInfo map;
        gst_buffer_map(buffer, &map, GST_MAP_WRITE);

        // Copy RGBA data
        for (int row = 0; row < height; row++) {
            std::memcpy(&map.data[row * videoWidth * 4],
                        &videoFrame.buffer[row * width * 4],
                        width * 4);
        }

        // Pack pose ID into the right side of the frame
        const pose_id_t poseIDToSend = videoFrame.poseID;
        packPoseIDIntoVideoFrame(poseIDToSend, map.data);

        gst_buffer_unmap(buffer, &map);
        stats.transferTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTransferTimeMs);

        double startEncode = timeutils::getTimeMicros();

        GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(appsrc), buffer);
        if (ret != GST_FLOW_OK) {
            spdlog::error("Failed to push buffer to GStreamer: {}", static_cast<int>(ret));
        }
        framesSent++;

        double frameEnd = timeutils::getTimeMicros();

        stats.encodeTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startEncode);

        stats.sendTimeMs = timeutils::microsToMillis(frameEnd - frameStart);

        double elapsedTimeSec = timeutils::microsToSeconds(frameEnd - frameStart);
        if (elapsedTimeSec < frameIntervalSec) {
            std::this_thread::sleep_for(std::chrono::microseconds(
                (int)(timeutils::secondsToMicros(frameIntervalSec - elapsedTimeSec))));
        }

        double now = timeutils::getTimeMicros();
        stats.totalSendTimeMs = timeutils::microsToMillis(now - prevTime);

        // Compute encoded bitrate in Mbps based on bytes metered at encoder output
        double elapsedSec = timeutils::microsToSeconds(now - prevTime);
        if (elapsedSec > 0.0) {
            uint64_t total = encodedBytesTotal.load(std::memory_order_relaxed);
            uint64_t deltaBytes = total - prevEncodedBytesTotal;
            prevEncodedBytesTotal = total;
            double bitsPerSec = static_cast<double>(deltaBytes) * 8.0 / elapsedSec;
            stats.bitrateMbps = bitsPerSec / BYTES_PER_MEGABYTE;
        }

        prevTime = now;
    }
}

