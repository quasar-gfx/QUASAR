#include <Utils/TimeUtils.h>

#include <DepthVideoTexture.h>

using namespace quasar;

DepthVideoTexture::DepthVideoTexture(const TextureDataCreateParams& params, std::string streamerURL)
    : DataReceiverTCP(streamerURL, false)
    , Texture(params)
{}

pose_id_t DepthVideoTexture::getLatestPoseID() {
    if (depthFrames.empty()) {
        return -1;
    }

    FrameData frameData = depthFrames.back();
    pose_id_t poseID = frameData.poseID;
    return poseID;
}

void DepthVideoTexture::onDataReceived(const std::vector<char>& data) {
    std::lock_guard<std::mutex> lock(m);

    std::vector<char> depthFrame = std::move(data);

    pose_id_t poseID;
    std::memcpy(&poseID, depthFrame.data(), sizeof(pose_id_t));

    depthFrame.erase(depthFrame.begin(), depthFrame.begin() + sizeof(pose_id_t));
    FrameData newFrameData = {poseID, std::move(depthFrame)};
    depthFrames.push_back(newFrameData);

    if (depthFrames.size() > maxQueueSize) {
        depthFrames.pop_front();
    }
}

pose_id_t DepthVideoTexture::draw(pose_id_t poseID) {
    std::lock_guard<std::mutex> lock(m);

    static float prevTime = timeutils::getTimeMicros();

    if (depthFrames.empty()) {
        return -1;
    }

    pose_id_t resPoseID = -1;
    std::vector<char> res;
    bool found = false;
    if (poseID == -1) {
        FrameData frameData = depthFrames.back();
        res = std::move(frameData.buffer);
        resPoseID = frameData.poseID;
        found = true;
    }
    else {
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

    int stride = width;
    glPixelStorei(GL_UNPACK_ROW_LENGTH, stride);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED, GL_UNSIGNED_SHORT, res.data());

    stats.timeToReceiveMs = timeutils::microsToMillis(timeutils::getTimeMicros() - prevTime);

    stats.bitrateMbps = ((sizeof(pose_id_t) + res.size() * 8) / timeutils::millisToSeconds(stats.timeToReceiveMs)) / BYTES_PER_MEGABYTE;

    prevPoseID = resPoseID;
    prevTime = timeutils::getTimeMicros();

    return resPoseID;
}
