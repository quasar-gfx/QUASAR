#include <cstring>
#include <spdlog/spdlog.h>
#include <Utils/FileIO.h>
#include <VideoTexture.h>

using namespace quasar;

#ifdef __ANDROID__
extern "C" {
    GST_PLUGIN_STATIC_DECLARE(app);
    GST_PLUGIN_STATIC_DECLARE(coreelements);
    GST_PLUGIN_STATIC_DECLARE(udp);
    GST_PLUGIN_STATIC_DECLARE(rtp);
    GST_PLUGIN_STATIC_DECLARE(rtpmanager);
    GST_PLUGIN_STATIC_DECLARE(libav);
    GST_PLUGIN_STATIC_DECLARE(videoconvertscale);
    GST_PLUGIN_STATIC_DECLARE(playback);
}
#endif

VideoTexture::VideoTexture(
        const TextureDataCreateParams& params,
        const std::string& videoURL)
    : videoURL(videoURL)
    , Texture(params)
{
    gst_init(nullptr, nullptr);

#ifdef __ANDROID__
    GST_PLUGIN_STATIC_REGISTER(app);
    GST_PLUGIN_STATIC_REGISTER(coreelements);
    GST_PLUGIN_STATIC_REGISTER(udp);
    GST_PLUGIN_STATIC_REGISTER(rtp);
    GST_PLUGIN_STATIC_REGISTER(rtpmanager);
    GST_PLUGIN_STATIC_REGISTER(libav);
    GST_PLUGIN_STATIC_REGISTER(videoconvertscale);
    GST_PLUGIN_STATIC_REGISTER(playback);
#endif

    // Parse host and port
    auto colonPos = videoURL.find(':');
    std::string host = videoURL.substr(0, colonPos);
    int port = std::stoi(videoURL.substr(colonPos + 1));

    std::string pipelineStr =
        "udpsrc name=udpsrc0 address=" + host + " port=" + std::to_string(port) +
        " caps=\"application/x-rtp,media=video,encoding-name=H264,payload=96\" ! "
        "rtpjitterbuffer ! rtph264depay ! avdec_h264 ! "
        "videoconvert ! video/x-raw,format=RGB ! appsink name=appsink0";

    GError* error = nullptr;
    pipeline = gst_parse_launch(pipelineStr.c_str(), &error);
    if (!pipeline || error) {
        spdlog::error("Failed to create GStreamer pipeline: {}", error ? error->message : "unknown");
        g_error_free(error);
        return;
    }

    appsink = gst_bin_get_by_name(GST_BIN(pipeline), "appsink0");
    gst_app_sink_set_emit_signals((GstAppSink*)appsink, false);
    gst_app_sink_set_drop((GstAppSink*)appsink, true);
    gst_app_sink_set_max_buffers((GstAppSink*)appsink, 1);

    GstElement* udpSrcElement = gst_bin_get_by_name(GST_BIN(pipeline), "udpsrc0");
    GstPad* srcPad = gst_element_get_static_pad(udpSrcElement, "src");
    gst_pad_add_probe(srcPad, GST_PAD_PROBE_TYPE_BUFFER,
        [](GstPad*, GstPadProbeInfo* info, gpointer userData) -> GstPadProbeReturn {
            auto* totalBytesRecv = static_cast<std::atomic<uint64_t>*>(userData);
            if (GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_BUFFER) {
                GstBuffer* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
                if (buffer) {
                    totalBytesRecv->fetch_add(gst_buffer_get_size(buffer));
                }
            }
            return GST_PAD_PROBE_OK;
        },
        &this->totalBytesRecv, nullptr);

    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    videoReceiverThread = std::thread(&VideoTexture::receiveFrame, this);
    spdlog::info("Created VideoTexture (GStreamer) that recvs from: {}", videoURL);
}

VideoTexture::~VideoTexture() {
    shouldTerminate = true;
    videoReady = false;

    if (videoReceiverThread.joinable())
        videoReceiverThread.join();

    if (appsink)
        gst_object_unref(appsink);
    if (pipeline)
        gst_object_unref(pipeline);
}

pose_id_t VideoTexture::getLatestPoseID() {
    std::lock_guard<std::mutex> lock(m);
    if (frames.empty()) return -1;
    return frames.back().poseID;
}

void VideoTexture::resize(uint width, uint height) {
    Texture::resize(width, height);
}

void VideoTexture::setMaxQueueSize(uint maxQueueSize) {
    this->maxQueueSize = maxQueueSize;
}

float VideoTexture::getFrameRate() {
    return 1.0f / timeutils::millisToSeconds(stats.totalTimeToRecvMs);
}

void VideoTexture::receiveFrame() {
    videoReady = true;

    time_t prevTime = timeutils::getTimeMicros();
    time_t lastBitrateCalcTime = 0;

    GstSample* sample = nullptr;
    GstBuffer* buffer = nullptr;
    GstMapInfo map;

    while (!shouldTerminate) {
        time_t frameStart = timeutils::getTimeMicros();

        // Get a sample from appsink
        sample = gst_app_sink_try_pull_sample(GST_APP_SINK(appsink), GST_SECOND / 2);
        if (!sample) continue;

        // Get video dimensions from the sample
        GstCaps* caps = gst_sample_get_caps(sample);
        GstStructure* s = gst_caps_get_structure(caps, 0);
        gst_structure_get_int(s, "width", (int*)&videoWidth);
        gst_structure_get_int(s, "height", (int*)&videoHeight);

        // Get the buffer from the sample
        buffer = gst_sample_get_buffer(sample);
        if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
            gst_sample_unref(sample);
            continue;
        }

        // Read the pose ID from the frame
        pose_id_t poseID = unpackPoseIDFromFrame(map.data, videoWidth, videoHeight);
        std::vector<char> frameData(map.data, map.data + map.size);

        {
            std::unique_lock<std::mutex> lock(m);
            frames.push_back({poseID, std::move(frameData)});
            if (frames.size() > maxQueueSize)
                frames.pop_front();
        }

        gst_buffer_unmap(buffer, &map);
        gst_sample_unref(sample);
        framesReceived++;

        time_t frameEnd = timeutils::getTimeMicros();

        stats.timeToReceiveMs = timeutils::microsToMillis(frameEnd - frameStart);

        // Bitrate calculation from pad probe
        time_t now = timeutils::getTimeMicros();
        time_t deltaSec = timeutils::microsToSeconds(now - lastBitrateCalcTime);
        if (deltaSec > 0.1) {
            stats.bitrateMbps = ((totalBytesRecv * 8.0) / BYTES_PER_MEGABYTE) / deltaSec;
            totalBytesRecv = 0;
            lastBitrateCalcTime = now;
        }

        stats.totalTimeToRecvMs = timeutils::microsToMillis(timeutils::getTimeMicros() - prevTime);
        prevTime = timeutils::getTimeMicros();
    }
}

pose_id_t VideoTexture::unpackPoseIDFromFrame(const uint8_t* data, int width, int height) {
    const int numVotes = 32;
    pose_id_t poseID = 0;

    for (int i = 0; i < poseIDOffset; i++) {
        int votes = 0;
        for (int j = 0; j < numVotes; j++) {
            int index = j * width * 3 + (width - 1 - i) * 3;
            uint8_t value = data[index];  // Red channel
            if (value > 127)
                votes++;
        }
        poseID |= (votes > numVotes / 2) << i;
    }

    return poseID;
}

pose_id_t VideoTexture::draw(pose_id_t poseID) {
    std::lock_guard<std::mutex> lock(m);
    if (!videoReady || frames.empty())
        return -1;

    FrameData& frameData = frames.back();
    if (poseID != -1) {
        for (auto& f : frames) {
            if (f.poseID == poseID) {
                frameData = f;
                break;
            }
        }
    }

    glPixelStorei(GL_UNPACK_ROW_LENGTH, videoWidth);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, frameData.buffer.data());

    prevPoseID = frameData.poseID;
    return frameData.poseID;
}
