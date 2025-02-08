#include <spdlog/spdlog.h>

#include <Utils/TimeUtils.h>
#include <BC4DepthVideoTexture.h>

#define BLOCK_SIZE 8

BC4DepthVideoTexture::BC4DepthVideoTexture(const TextureDataCreateParams &params, std::string streamerURL)
        : streamerURL(streamerURL)
        , DataReceiverTCP(streamerURL, false)
        , Texture(params) {
    // round up to nearest multiple of BLOCK_SIZE
    width = (params.width + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    height = (params.height + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    resize(width, height);

    compressedSize = (width / BLOCK_SIZE) * (height / BLOCK_SIZE);
    bc4CompressedBuffer = Buffer(GL_SHADER_STORAGE_BUFFER, compressedSize, sizeof(Block), nullptr, GL_DYNAMIC_DRAW);
}

pose_id_t BC4DepthVideoTexture::getLatestPoseID() {
    std::lock_guard<std::mutex> lock(m);
    if (depthFrames.empty()) {
        return -1;
    }

    FrameData frameData = depthFrames.back();
    return frameData.poseID;
}

void BC4DepthVideoTexture::onDataReceived(const std::vector<char>& compressedData) {
    static float prevTime = timeutils::getTimeMicros();

    float startTime = timeutils::getTimeMicros();

    // calculate expected decompressed size
    size_t expectedSize = sizeof(pose_id_t) + compressedSize * sizeof(Block);
    std::vector<char> decompressedData(expectedSize);

    // decompress in one shot
    size_t dstSize = compressor.decompress(compressedData, decompressedData);

    stats.timeToDecompressMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
    stats.compressionRatio = static_cast<float>(dstSize) / compressedData.size();

    // extract pose ID
    pose_id_t poseID;
    std::memcpy(&poseID, decompressedData.data(), sizeof(pose_id_t));

    // create frame data with the rest of the buffer
    std::vector<char> frameBuffer(decompressedData.begin() + sizeof(pose_id_t),
                                     decompressedData.end());

    std::lock_guard<std::mutex> lock(m);

    FrameData newFrameData = {poseID, std::move(frameBuffer)};
    depthFrames.push_back(newFrameData);

    if (depthFrames.size() > maxQueueSize) {
        depthFrames.pop_front();
    }

    stats.timeToReceiveMs = timeutils::microsToMillis(timeutils::getTimeMicros() - prevTime);
    stats.bitrateMbps = ((sizeof(pose_id_t) + compressedData.size()) * 8 / timeutils::millisToSeconds(stats.timeToReceiveMs)) / BYTES_IN_MB;

    prevTime = timeutils::getTimeMicros();
}

pose_id_t BC4DepthVideoTexture::draw(pose_id_t poseID) {
    std::lock_guard<std::mutex> lock(m);

    if (depthFrames.empty()) {
        return -1;
    }

    pose_id_t resPoseID = -1;
    std::vector<char> res;
    if (poseID == -1) {
        FrameData frameData = depthFrames.back();
        res = std::move(frameData.buffer);
        resPoseID = frameData.poseID;
    }
    else {
        for (auto it = depthFrames.begin(); it != depthFrames.end(); ++it) {
            FrameData frameData = *it;
            if (frameData.poseID == poseID) {
                res = std::move(frameData.buffer);
                resPoseID = frameData.poseID;
                break;
            }
        }

        if (res.empty()) {
            return -1;
        }
    }

    // update the BC4 compressed buffer
    bc4CompressedBuffer.setData(compressedSize, res.data());

    prevPoseID = resPoseID;

    return resPoseID;
}
