#ifndef VIDEO_TEXTURE_H
#define VIDEO_TEXTURE_H

#include <gst/gst.h>
#include <gst/app/gstappsink.h>

#include <deque>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <Utils/TimeUtils.h>
#include <Texture.h>
#include <CameraPose.h>

namespace quasar {

class VideoTexture : public Texture {
public:
    std::string videoURL = "127.0.0.1:12345";

    struct Stats {
        double timeToReceiveMs = 0.0;
        double totalTimeToRecvMs = 0.0;
        double bitrateMbps = 0.0;
    } stats;

    uint videoWidth, videoHeight;

    VideoTexture(
        const TextureDataCreateParams& params,
        const std::string& videoURL);
    ~VideoTexture();

    void resize(uint width, uint height);

    pose_id_t getLatestPoseID();
    float getFrameRate();

    void setMaxQueueSize(uint maxQueueSize);

    pose_id_t draw(pose_id_t poseID = -1);

private:
    pose_id_t prevPoseID = -1;
    uint64_t framesReceived = 0;
    uint maxQueueSize = 10;

    mutable std::atomic<uint64_t> totalBytesRecv = 0;

    int poseIDOffset = sizeof(pose_id_t) * 8;

    std::atomic_bool videoReady = false;
    bool shouldTerminate = false;

    std::thread videoReceiverThread;
    std::mutex m;
    std::condition_variable cv;

    struct FrameData {
        pose_id_t poseID;
        std::vector<char> buffer; // raw RGB frame
    };
    std::deque<FrameData> frames;

    pose_id_t unpackPoseIDFromFrame(const uint8_t* data, int width, int height);

    void receiveFrame();

    // GStreamer components
    GstElement* pipeline = nullptr;
    GstElement* appsink = nullptr;
};

} // namespace quasar

#endif // VIDEO_TEXTURE_H

