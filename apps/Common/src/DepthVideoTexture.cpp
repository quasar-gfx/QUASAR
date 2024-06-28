#include <Utils/TimeUtils.h>

#include <DepthVideoTexture.h>

pose_id_t DepthVideoTexture::draw(pose_id_t poseID) {
    static float prevTime = timeutils::getCurrTimeMicros();\

    std::vector<uint8_t> data = receiver.recv();
    if (data.empty()) {
        prevTime = timeutils::getCurrTimeMicros();
        return prevPoseID;
    }

    pose_id_t pID = *reinterpret_cast<pose_id_t*>(data.data());
    data.erase(data.begin(), data.begin() + sizeof(pose_id_t));
    FrameData newFrameData = {pID, std::move(data)};
    datas.push_back(newFrameData);

    if (datas.size() > maxQueueSize) {
        datas.pop_front();
    }

    pose_id_t resPoseID = -1;
    std::vector<uint8_t> res;
    bool found = false;
    if (poseID == -1) {
        FrameData frameData = datas.back();
        res = std::move(frameData.buffer);
        resPoseID = frameData.poseID;
        found = true;
    }
    else {
        for (auto it = datas.begin(); it != datas.end(); ++it) {
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
        prevTime = timeutils::getCurrTimeMicros();
        return prevPoseID;
    }

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED, GL_UNSIGNED_SHORT, res.data());

    stats.timeToReceiveMs = timeutils::microsToMillis(timeutils::getCurrTimeMicros() - prevTime);

    stats.bitrateMbps = ((sizeof(pose_id_t) + res.size() * 8) / timeutils::millisToSeconds(stats.timeToReceiveMs)) / MBPS_TO_BPS;

    prevPoseID = resPoseID;
    prevTime = timeutils::getCurrTimeMicros();

    return resPoseID;
}
