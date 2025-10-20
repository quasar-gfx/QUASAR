#include <sstream>
#include <spdlog/spdlog.h>
#include <Receivers/VideoTexture.h>
#include <Networking/Utils.h>

using namespace quasar;

#ifdef PLATFORM_ANDROID
extern "C" {
    static JavaVM* _java_vm;
    static jobject _app_context;
    static jobject _class_loader;

    /* AMC plugin uses needs these defined */
    JavaVM* gst_android_get_java_vm(void) { return _java_vm; }
    jobject gst_android_get_application_context(void) { return _app_context; }
    jobject gst_android_get_application_class_loader(void) { return _class_loader; }

    GST_PLUGIN_STATIC_DECLARE(coreelements);
    GST_PLUGIN_STATIC_DECLARE(app);
    GST_PLUGIN_STATIC_DECLARE(videoconvertscale);
    GST_PLUGIN_STATIC_DECLARE(videorate);
    GST_PLUGIN_STATIC_DECLARE(rtp);
    GST_PLUGIN_STATIC_DECLARE(rtpmanager);
    GST_PLUGIN_STATIC_DECLARE(udp);
    GST_PLUGIN_STATIC_DECLARE(srt);
    GST_PLUGIN_STATIC_DECLARE(mpegtsdemux);
    GST_PLUGIN_STATIC_DECLARE(playback);
    GST_PLUGIN_STATIC_DECLARE(androidmedia);
    GST_PLUGIN_STATIC_DECLARE(videoparsersbad);
}

void VideoTexture::gst_android_glue_init(ANativeActivity* activity) {
    JNIEnv* env = nullptr;
    activity->vm->AttachCurrentThread(&env, nullptr);

    // Save VM
    _java_vm = activity->vm;

    // Application Context
    jclass activityClass = env->GetObjectClass(activity->clazz);
    jmethodID getAppContext = env->GetMethodID(activityClass, "getApplicationContext", "()Landroid/content/Context;");
    jobject appContextLocal = env->CallObjectMethod(activity->clazz, getAppContext);
    _app_context = env->NewGlobalRef(appContextLocal);
    env->DeleteLocalRef(appContextLocal);

    // ClassLoader from Context
    jclass contextCls = env->GetObjectClass(_app_context);
    jmethodID getCL = env->GetMethodID(contextCls, "getClassLoader", "()Ljava/lang/ClassLoader;");
    jobject clLocal = env->CallObjectMethod(_app_context, getCL);
    _class_loader = env->NewGlobalRef(clLocal);
    env->DeleteLocalRef(clLocal);

    env->DeleteLocalRef(contextCls);
    env->DeleteLocalRef(activityClass);
}
#endif

VideoTexture::VideoTexture(
        const TextureDataCreateParams& params,
        const std::string& videoURL,
        bool useRTP)
    : videoURL(videoURL)
    , Texture(params)
{
    if (videoURL.empty()) {
        return;
    }

    gst_init(nullptr, nullptr);
#ifdef PLATFORM_ANDROID
    GST_PLUGIN_STATIC_REGISTER(coreelements);
    GST_PLUGIN_STATIC_REGISTER(app);
    GST_PLUGIN_STATIC_REGISTER(videoconvertscale);
    GST_PLUGIN_STATIC_REGISTER(videorate);
    GST_PLUGIN_STATIC_REGISTER(rtp);
    GST_PLUGIN_STATIC_REGISTER(rtpmanager);
    GST_PLUGIN_STATIC_REGISTER(udp);
    GST_PLUGIN_STATIC_REGISTER(srt);
    GST_PLUGIN_STATIC_REGISTER(mpegtsdemux);
    GST_PLUGIN_STATIC_REGISTER(playback);
    GST_PLUGIN_STATIC_REGISTER(androidmedia);
    GST_PLUGIN_STATIC_REGISTER(videoparsersbad);
#endif

    GstRegistry* registry = gst_registry_get();
    GList* factories = gst_registry_get_feature_list(registry, GST_TYPE_ELEMENT_FACTORY);
    std::ostringstream codecs;
    for (GList* l = factories; l != nullptr; l = l->next) {
        GstElementFactory* factory = GST_ELEMENT_FACTORY(l->data);
        const gchar* klass = gst_element_factory_get_metadata(factory, GST_ELEMENT_METADATA_KLASS);
        const gchar* name  = gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(factory));
        if (klass && g_strrstr(klass, "Decoder")) {
            codecs << name << " ";
        }
    }
    spdlog::debug("Available Decoders: {}", codecs.str());
    gst_plugin_feature_list_free(factories);

    // Parse host and port
    auto [host, port] = networkutils::parseIPAddressAndPort(videoURL);

#ifndef PLATFORM_ANDROID
    std::string decoderName = "avdec_h264";
#else
    std::string decoderName = "decodebin"; // decodebin should automatically select a hardware decoder
#endif

    std::ostringstream oss;
    if (useRTP) {
        oss << "udpsrc name=" << srcName
            << " address=" << host
            << " port=" << port
            << " caps=\"application/x-rtp, media=video, encoding-name=H264, payload=96, clock-rate=90000\" ! "
            << "rtpjitterbuffer ! "
            << "rtph264depay ! ";
    }
    else {
        oss << "srtsrc name=" << srcName
            << " uri=\"srt://" << host << ":" << port << "?mode=listener&latency=80\" ! ";
    }
    oss << "h264parse ! "
        << decoderName << " ! "
        << "videoconvert ! video/x-raw,format=RGB ! "
        << "queue leaky=downstream max-size-buffers=1 max-size-time=0 max-size-bytes=0 ! "
        << "appsink name=" << appSinkName
        << " sync=false max-buffers=1 drop=true";
    std::string pipelineStr = oss.str();

    GError* error = nullptr;
    pipeline = gst_parse_launch(pipelineStr.c_str(), &error);
    if (!pipeline || error) {
        spdlog::error("Failed to create GStreamer pipeline: {}", error ? error->message : "unknown");
        if (error) g_error_free(error);
        return;
    }

    appsink = gst_bin_get_by_name(GST_BIN(pipeline), appSinkName.c_str());
    gst_app_sink_set_emit_signals((GstAppSink*)appsink, false);
    gst_app_sink_set_drop((GstAppSink*)appsink, true);
    gst_app_sink_set_max_buffers((GstAppSink*)appsink, 1);

    GstElement* udpSrcElement = gst_bin_get_by_name(GST_BIN(pipeline), srcName.c_str());
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
    gst_object_unref(srcPad);
    gst_object_unref(udpSrcElement);

    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    videoReceiverThread = std::thread(&VideoTexture::receiveFrame, this);
    spdlog::info("Created VideoTexture (GStreamer) that recvs from: {}://{}", useRTP ? "rtp" : "srt", videoURL);
}

VideoTexture::~VideoTexture() {
    stop();
}

void VideoTexture::stop() {
    shouldTerminate = true;

    if (pipeline)
        gst_element_set_state(pipeline, GST_STATE_NULL);

    if (videoReceiverThread.joinable())
        videoReceiverThread.join();

    if (appsink)
        gst_object_unref(appsink);
    if (pipeline)
        gst_object_unref(pipeline);

    appsink = nullptr;
    pipeline = nullptr;
}

void VideoTexture::resize(uint width, uint height) {
    Texture::resize(width, height);
}

void VideoTexture::receiveFrame() {
    double prevTime = timeutils::getTimeMicros();
    double lastBitrateCalcTime = 0;

    GstSample* sample = nullptr;
    GstBuffer* buffer = nullptr;
    GstMapInfo map;

    while (!shouldTerminate) {
        double frameStart = timeutils::getTimeMicros();

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

        {
            std::unique_lock<std::mutex> lock(m);
            if (frames.size() >= maxQueueSize) {
                FrameData frame = std::move(frames.front());
                frames.pop_front();
                frame.poseID = poseID;
                frame.buffer.resize(map.size);
                std::memcpy(frame.buffer.data(), map.data, map.size);
                frames.push_back(std::move(frame));
            }
            else {
                FrameData frame;
                frame.poseID = poseID;
                frame.buffer.resize(map.size);
                std::memcpy(frame.buffer.data(), map.data, map.size);
                frames.push_back(std::move(frame));
            }
        }

        gst_buffer_unmap(buffer, &map);
        gst_sample_unref(sample);
        framesReceived++;

        double frameEnd = timeutils::getTimeMicros();

        stats.receiveTimeMs = timeutils::microsToMillis(frameEnd - frameStart);

        // Bitrate calculation from pad probe
        double now = timeutils::getTimeMicros();
        double deltaSec = timeutils::microsToSeconds(now - lastBitrateCalcTime);
        if (deltaSec > 0.1) {
            stats.bitrateMbps = ((totalBytesRecv * 8.0) / BYTES_PER_MEGABYTE) / deltaSec;
            totalBytesRecv = 0;
            lastBitrateCalcTime = now;
        }

        stats.totalRecvTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - prevTime);
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
            uint8_t value = data[index]; // Red channel
            if (value > 127)
                votes++;
        }
        poseID |= (votes > numVotes / 2) << i;
    }

    return poseID;
}

pose_id_t VideoTexture::getLatestPoseID() {
    std::lock_guard<std::mutex> lock(m);
    if (frames.empty()) return -1;
    return frames.back().poseID;
}

bool VideoTexture::containsFrames() {
    std::lock_guard<std::mutex> lock(m);
    return !frames.empty();
}

bool VideoTexture::containsFrameWithPoseID(pose_id_t poseID) {
    std::lock_guard<std::mutex> lock(m);
    for (const auto& f : frames) {
        if (f.poseID == poseID) {
            return true;
        }
    }
    return false;
}

pose_id_t VideoTexture::draw(pose_id_t poseID) {
    std::lock_guard<std::mutex> lock(m);
    if (frames.empty()) {
        return -1;
    }

    if (poseID != -1 && poseID == prevPoseID) {
        return prevPoseID;
    }

    FrameData* frameData = nullptr;
    if (poseID != -1) {
        for (auto& f : frames) {
            if (f.poseID == poseID) {
                frameData = &f;
                break;
            }
        }
    }
    else {
        frameData = &frames.back();
    }

    if (frameData == nullptr) {
        return prevPoseID;
    }

    glPixelStorei(GL_UNPACK_ROW_LENGTH, videoWidth);
    loadFromData(frameData->buffer.data(), false);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);

    prevPoseID = frameData->poseID;
    return frameData->poseID;
}
