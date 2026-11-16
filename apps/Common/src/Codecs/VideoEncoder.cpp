#include <algorithm>
#include <vector>

#include <gst/app/gstappsrc.h>
#include <spdlog/spdlog.h>

#include <Codecs/VideoEncoder.h>

namespace {

GstElement* makeEncoder(const char* name) {
    GstElementFactory* factory = gst_element_factory_find(name);
    if (!factory) return nullptr;
    GstElement* element = gst_element_factory_create(factory, nullptr);
    gst_object_unref(factory);
    return element;
}

std::vector<std::string> getVideoEncoders(const quasar::VideoEncoderParams& params) {
    const std::string gop = std::to_string(std::max(1u, params.framerate));
    const std::string bitrate = std::to_string(params.bitrate);
    std::vector<std::string> encoders;

    if (GstElement* element = makeEncoder("nvcudah264enc")) {
        auto* elementClass = G_OBJECT_GET_CLASS(element);
        std::string encoder = "videoconvert ! video/x-raw,format=NV12 ! nvcudah264enc name=encoder"
            " preset=p1 tune=ultra-low-latency zero-reorder-delay=true gop-size=" + gop;
        if (g_object_class_find_property(elementClass, "b-frames")) encoder += " b-frames=0";
        if (g_object_class_find_property(elementClass, "rc-lookahead")) encoder += " rc-lookahead=0";
        encoder += " rate-control=cbr bitrate=" + bitrate;
        encoders.push_back(encoder);
        gst_object_unref(element);
    }
    if (GstElement* element = makeEncoder("nvh264enc")) {
        auto* elementClass = G_OBJECT_GET_CLASS(element);
        std::string encoder = "videoconvert ! video/x-raw,format=NV12 ! nvh264enc name=encoder";
        GParamSpec* preset = g_object_class_find_property(elementClass, "preset");
        if (preset && G_IS_PARAM_SPEC_ENUM(preset)) {
            auto* values = static_cast<GEnumClass*>(g_type_class_ref(G_PARAM_SPEC_VALUE_TYPE(preset)));
            for (const char* name : { "p1", "hp", "default" }) {
                if (g_enum_get_value_by_nick(values, name)) {
                    encoder += std::string(" preset=") + name;
                    break;
                }
            }
            g_type_class_unref(values);
        }
        if (g_object_class_find_property(elementClass, "rate-control")) {
            encoder += " zero-reorder-delay=true";
            if (g_object_class_find_property(elementClass, "b-frames")) encoder += " b-frames=0";
            if (g_object_class_find_property(elementClass, "rc-lookahead")) encoder += " rc-lookahead=0";
            encoder += " rate-control=cbr bitrate=" + bitrate;
            if (g_object_class_find_property(elementClass, "tune")) encoder += " tune=ultra-low-latency";
        }
        else {
            encoder += " zerolatency=true";
            if (g_object_class_find_property(elementClass, "bframes")) encoder += " bframes=0";
            encoder += " rc-mode=cbr bitrate=" + bitrate;
        }
        encoders.push_back(encoder + " gop-size=" + gop);
        gst_object_unref(element);
    }
    if (GstElement* element = makeEncoder("x264enc")) {
        encoders.push_back("videoconvert ! video/x-raw,format=I420 ! x264enc name=encoder"
            " speed-preset=ultrafast tune=zerolatency byte-stream=true bframes=0 key-int-max=" +
            gop + " bitrate=" + bitrate);
        gst_object_unref(element);
    }
    return encoders;
}

GstPadProbeReturn firstEncodedBuffer(GstPad* pad, GstPadProbeInfo* info, gpointer) {
    GstBuffer* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
    if (GST_BUFFER_FLAG_IS_SET(buffer, GST_BUFFER_FLAG_HEADER) && !GST_BUFFER_PTS_IS_VALID(buffer)) {
        return GST_PAD_PROBE_DROP;
    }
    GstElement* parser = gst_pad_get_parent_element(pad);
    gst_element_post_message(parser, gst_message_new_application(GST_OBJECT(parser),
        gst_structure_new_empty("quasar-video-started")));
    gst_object_unref(parser);
    return GST_PAD_PROBE_REMOVE;
}

} // namespace

GstElement* quasar::startVideoPipeline(const VideoEncoderParams& params, const std::string& destination,
                                     GstBuffer* firstFrame) {
    if (!gst_is_initialized()) gst_init(nullptr, nullptr);
    const std::string source = "appsrc name=src is-live=true format=time do-timestamp=true "
        "caps=video/x-raw,format=RGBA,width=" + std::to_string(params.width) +
        ",height=" + std::to_string(params.height) +
        ",framerate=" + std::to_string(std::max(1u, params.framerate)) + "/1 ! "
        "queue leaky=upstream max-size-buffers=1 max-size-time=0 max-size-bytes=0 ! ";

    for (const auto& encoder : getVideoEncoders(params)) {
        const std::string description = source + encoder +
            " ! h264parse name=parser config-interval=1 ! " + destination;
        GError* error = nullptr;
        GstElement* pipeline = gst_parse_launch(description.c_str(), &error);
        if (error || !pipeline) {
            spdlog::warn("Video encoder unavailable: {}", error ? error->message : "no pipeline");
            g_clear_error(&error);
            if (pipeline) gst_object_unref(pipeline);
            continue;
        }
        GstElement* src = gst_bin_get_by_name(GST_BIN(pipeline), "src");
        GstElement* parser = gst_bin_get_by_name(GST_BIN(pipeline), "parser");
        GstPad* pad = gst_element_get_static_pad(parser, "src");
        GstBus* bus = gst_element_get_bus(pipeline);
        gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER, firstEncodedBuffer, nullptr, nullptr);
        bool started = gst_element_set_state(pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE;
        if (started) started = gst_app_src_push_buffer(GST_APP_SRC(src), gst_buffer_ref(firstFrame)) == GST_FLOW_OK;

        // NVENC can fail asynchronously when the first frame opens the session.
        const GstClockTime deadline = gst_util_get_timestamp() + 2 * GST_SECOND;
        GstMessage* message = nullptr;
        bool ready = false;
        while (started) {
            const GstClockTime now = gst_util_get_timestamp();
            message = gst_bus_timed_pop_filtered(bus, now < deadline ? deadline - now : 0,
                static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_APPLICATION | GST_MESSAGE_EOS));
            if (!message || GST_MESSAGE_TYPE(message) != GST_MESSAGE_APPLICATION) break;
            ready = gst_structure_has_name(gst_message_get_structure(message), "quasar-video-started");
            gst_message_unref(message);
            message = nullptr;
            if (ready || now >= deadline) break;
        }
        if (!message) message = gst_bus_pop_filtered(bus, GST_MESSAGE_ERROR);
        if (message) {
            if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR) {
                gchar* debug = nullptr;
                gst_message_parse_error(message, &error, &debug);
                spdlog::warn("Video encoder failed: {} ({})", error->message, debug ? debug : "no details");
                g_clear_error(&error);
                g_free(debug);
            }
            gst_message_unref(message);
            ready = false;
        }
        if (!ready) {
            spdlog::warn("Video encoder could not start: {}", encoder);
            gst_element_set_state(pipeline, GST_STATE_NULL);
        }
        gst_object_unref(bus);
        gst_object_unref(pad);
        gst_object_unref(parser);
        gst_object_unref(src);
        if (ready) {
            spdlog::info("Video encoder: {} ({}x{})", encoder, params.width, params.height);
            return pipeline;
        }
        gst_object_unref(pipeline);
    }
    spdlog::error("No working H.264 encoder is available");
    return nullptr;
}
