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
    std::string videoURL = "0.0.0.0:12345";

    struct Stats {
        double receiveTimeMs = 0.0;
        double totalRecvTimeMs = 0.0;
        double bitrateMbps = 0.0;
    } stats;

    uint videoWidth, videoHeight;

    VideoTexture(
        const TextureDataCreateParams& params,
        const std::string& videoURL,
        bool useSRT = false);
    ~VideoTexture();

    void stop();

    pose_id_t getLatestPoseID();

    float getFrameRate() { return 1.0f / timeutils::millisToSeconds(stats.totalRecvTimeMs); }

    void setMaxQueueSize(size_t maxQueueSize) { this->maxQueueSize = maxQueueSize; }

    bool containsFrames();
    bool containsFrameWithPoseID(pose_id_t poseID);
    pose_id_t draw(pose_id_t poseID = -1);

    void resize(uint width, uint height);

#ifdef __ANDROID__
    // Registers Android JNI with GStreamer
    static void gst_android_glue_init(ANativeActivity* activity);
#endif

private:
    pose_id_t prevPoseID = -1;
    uint64_t framesReceived = 0;
    size_t maxQueueSize = 3;

    const int poseIDOffset = sizeof(pose_id_t) * 8;

    std::atomic_bool shouldTerminate = false;

    std::string srcName = "src0";
    std::string appSinkName = "appsink0";

    mutable std::atomic<uint64_t> totalBytesRecv = 0;

    std::thread videoReceiverThread;
    std::mutex m;
    std::condition_variable cv;

    struct FrameData {
        pose_id_t poseID;
        std::vector<char> buffer; // raw RGB frame
    };
    std::deque<FrameData> frames;

    GstElement* pipeline = nullptr;
    GstElement* appsink = nullptr;

    pose_id_t unpackPoseIDFromFrame(const uint8_t* data, int width, int height);

    void receiveFrame();
};

} // namespace quasar

#endif // VIDEO_TEXTURE_H
