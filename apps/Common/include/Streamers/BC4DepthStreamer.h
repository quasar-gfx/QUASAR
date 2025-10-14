#ifndef BC4_DEPTH_STREAMER_H
#define BC4_DEPTH_STREAMER_H

#include <iomanip>
#include <thread>
#include <atomic>

#include <concurrentqueue/concurrentqueue.h>

#include <Shaders/ComputeShader.h>
#include <RenderTargets/RenderTarget.h>
#include <Networking/DataStreamerTCP.h>
#include <Receivers/BC4DepthVideoTexture.h>

#include <Path.h>
#include <Codecs/BC4.h>
#include <Codecs/ZSTDCodec.h>
#include <CameraPose.h>

namespace quasar {

class BC4DepthStreamer : public RenderTarget, public DataStreamerTCP {
public:
    uint width, height;

    Buffer bc4CompressedBuffer;

    std::string receiverURL;
    size_t compressedSize;

    struct Stats {
        double transferTimeMs = 0.0;
        double compressTimeMs = 0.0;
        double sendTimeMs = 0.0;
        double bitrateMbps = 0.0;
        double compressionRatio = 0.0;
    } stats;

    BC4DepthStreamer(const RenderTargetCreateParams& params, const std::string& receiverURL = "", uint maxFrameRate = 30);
    ~BC4DepthStreamer() = default;

    float getFrameRate() const { return 1.0f / timeutils::millisToSeconds(stats.sendTimeMs); }

    size_t generateFrame();
    void sendFrame(pose_id_t poseID);
    size_t writeToFile(const Path& filename);
    size_t writeToMemory(pose_id_t poseID, std::vector<char>& outputData);

private:
    uint maxFrameRate;

    double prevTime;

    std::vector<char> dataBC4;
    std::vector<char> dataZSTD;
    std::vector<char> compressedData;

    ZSTDCodec codec;
    ComputeShader bc4CompressionShader;
};

} // namespace quasar

#endif // BC4_DEPTH_STREAMER_H
