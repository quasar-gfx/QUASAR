#include <Utils/TimeUtils.h>
#include <BC4DepthVideoTexture.h>

BC4DepthVideoTexture::BC4DepthVideoTexture(const TextureDataCreateParams &params, std::string streamerURL)
    : streamerURL(streamerURL)
    , DataReceiverTCP(streamerURL, false)
    , Texture(params) {

    compressedSize = (params.width / 8) * (params.height / 8) * sizeof(Block);
    bc4CompressedBuffer = Buffer<Block>(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW, compressedSize / sizeof(Block), nullptr);
}

pose_id_t BC4DepthVideoTexture::getLatestPoseID() {
    std::lock_guard<std::mutex> lock(m);
    if (depthFrames.empty()) {
        return -1;
    }
    FrameData frameData = depthFrames.back();
    return frameData.poseID;
}

void BC4DepthVideoTexture::onDataReceived(const std::vector<uint8_t>& data) {
    std::lock_guard<std::mutex> lock(m);

    if (data.size() != sizeof(pose_id_t) + compressedSize) {
        std::cerr << "Received data size mismatch. Expected: " << (sizeof(pose_id_t) + compressedSize) << ", Got: " << data.size() << std::endl;
        return;
    }

    // debugPrintData(data); // Add this line to print the received data

    std::vector<uint8_t> depthFrame = data;
    pose_id_t poseID;
    std::memcpy(&poseID, depthFrame.data(), sizeof(pose_id_t));
    depthFrame.erase(depthFrame.begin(), depthFrame.begin() + sizeof(pose_id_t));

    FrameData newFrameData = {poseID, std::move(depthFrame)};
    depthFrames.push_back(newFrameData);

    if (depthFrames.size() > maxQueueSize) {
        depthFrames.pop_front();
    }
}

pose_id_t BC4DepthVideoTexture::draw(pose_id_t poseID) {
    std::lock_guard<std::mutex> lock(m);
    static float prevTime = timeutils::getTimeMicros();

    if (depthFrames.empty()) {
        return -1;
    }

    pose_id_t resPoseID = -1;
    std::vector<uint8_t> res;
    bool found = false;

    if (poseID == -1) {
        FrameData frameData = depthFrames.back();
        res = std::move(frameData.buffer);
        resPoseID = frameData.poseID;
        found = true;
    } else {
        for (auto it = depthFrames.begin(); it != depthFrames.end(); ++it) {
            FrameData frameData = *it;
            if (frameData.poseID == poseID) {
                res = std::move(frameData.buffer);
                resPoseID = frameData.poseID;
                found = true;
                break;
            }
        }
    }

    if (!found) {
        prevTime = timeutils::getTimeMicros();
        return prevPoseID;
    }

    // Update the BC4 compressed buffer
    bc4CompressedBuffer.setData(compressedSize / sizeof(Block), res.data());

    stats.timeToReceiveMs = timeutils::microsToMillis(timeutils::getTimeMicros() - prevTime);
    stats.bitrateMbps = ((sizeof(pose_id_t) + res.size() * 8) / timeutils::millisToSeconds(stats.timeToReceiveMs)) / MBPS_TO_BPS;

    prevPoseID = resPoseID;
    prevTime = timeutils::getTimeMicros();

    return resPoseID;
}
