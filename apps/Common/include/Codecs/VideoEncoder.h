#ifndef VIDEO_ENCODER_H
#define VIDEO_ENCODER_H

#include <cstdint>
#include <string>

#include <gst/gst.h>

namespace quasar {

struct VideoEncoderParams {
    uint32_t width;
    uint32_t height;
    uint32_t framerate;
    uint32_t bitrate;
};

GstElement* startVideoPipeline(const VideoEncoderParams& params, const std::string& destination,
                              GstBuffer* firstFrame);

} // namespace quasar

#endif // VIDEO_ENCODER_H
