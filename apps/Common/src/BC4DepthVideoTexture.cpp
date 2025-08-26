#include <spdlog/spdlog.h>

#include <Utils/TimeUtils.h>
#include <BC4DepthVideoTexture.h>

using namespace quasar;

BC4DepthVideoTexture::BC4DepthVideoTexture(const TextureDataCreateParams& params, std::string streamerURL)
    : DataReceiverTCP(streamerURL, false)
    , width((params.width + BC4_BLOCK_SIZE - 1) / BC4_BLOCK_SIZE * BC4_BLOCK_SIZE) // Round up to nearest multiple of BC4_BLOCK_SIZE
    , height((params.height + BC4_BLOCK_SIZE - 1) / BC4_BLOCK_SIZE * BC4_BLOCK_SIZE)
    , compressedSize((width / BC4_BLOCK_SIZE) * (height / BC4_BLOCK_SIZE))
    , Texture(params)
{
    resize(width, height);
    bc4CompressedBuffer = Buffer({
        .target = GL_SHADER_STORAGE_BUFFER,
        .dataSize = sizeof(BC4Block),
        .numElems = compressedSize,
        .usage = GL_DYNAMIC_DRAW,
    });
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
    time_t startTime = timeutils::getTimeMicros();

    // Calculate expected decompressed size
    size_t expectedSize = compressedSize * sizeof(BC4Block);
    std::vector<char> decompressedData(expectedSize);

    // Decompress in one shot
    size_t decompressedSize = codec.decompress(compressedData, decompressedData);

    stats.timeToDecompressMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
    stats.compressionRatio = static_cast<float>(decompressedSize) / compressedData.size();

    // Extract pose ID
    pose_id_t poseID;
    std::memcpy(&poseID, decompressedData.data(), sizeof(pose_id_t));

    // Create frame data with the rest of the buffer
    std::vector<char> frameBuffer(decompressedData.begin() + sizeof(pose_id_t), decompressedData.end());

    std::lock_guard<std::mutex> lock(m);

    FrameData newFrameData = {poseID, std::move(frameBuffer)};
    depthFrames.push_back(newFrameData);

    if (depthFrames.size() > maxQueueSize) {
        depthFrames.pop_front();
    }

    stats.timeToReceiveMs = timeutils::microsToMillis(timeutils::getTimeMicros() - prevTime);
    stats.bitrateMbps = ((compressedData.size() * 8) / timeutils::millisToSeconds(stats.timeToReceiveMs)) / BYTES_PER_MEGABYTE;

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

    // Update the BC4 compressed buffer
    bc4CompressedBuffer.bind();
    void* dst = bc4CompressedBuffer.mapToCPU(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_UNSYNCHRONIZED_BIT);
    if (dst) {
        std::memcpy(dst, res.data(), res.size());
        bc4CompressedBuffer.unmapFromCPU();
    }
    else {
        spdlog::warn("Failed to map BC4 compressed buffer. Copying using setData");
        bc4CompressedBuffer.setData(res.size(), res.data());
    }
    bc4CompressedBuffer.unbind();

    prevPoseID = resPoseID;

    return resPoseID;
}
