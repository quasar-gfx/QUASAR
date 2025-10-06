#include <spdlog/spdlog.h>

#include <Utils/TimeUtils.h>
#include <Receivers/BC4DepthVideoTexture.h>

using namespace quasar;

BC4DepthVideoTexture::BC4DepthVideoTexture(const TextureDataCreateParams& params, std::string streamerURL)
    : width(((params.width + BC4_BLOCK_SIZE - 1) / BC4_BLOCK_SIZE) * BC4_BLOCK_SIZE) // Round up to nearest multiple of BC4_BLOCK_SIZE
    , height(((params.height + BC4_BLOCK_SIZE - 1) / BC4_BLOCK_SIZE) * BC4_BLOCK_SIZE)
    , compressedSize((width / BC4_BLOCK_SIZE) * (height / BC4_BLOCK_SIZE))
    , Texture(params)
    , DataReceiverTCP(streamerURL, false)
{
    resize(width, height);
    bc4CompressedBuffer = Buffer({
        .target = GL_SHADER_STORAGE_BUFFER,
        .dataSize = sizeof(BC4Block),
        .numElems = compressedSize,
        .usage = GL_DYNAMIC_DRAW,
    });

    // Calculate max decompressed size
    decompressedData.resize(compressedSize * sizeof(BC4Block));

    if (!streamerURL.empty()) {
        spdlog::info("Created BC4DepthVideoTexture that recvs from URL: tcp://{}", streamerURL);
    }
}

pose_id_t BC4DepthVideoTexture::getLatestPoseID() {
    std::lock_guard<std::mutex> lock(m);
    if (frames.empty()) {
        return -1;
    }

    FrameData frameData = frames.back();
    return frameData.poseID;
}

void BC4DepthVideoTexture::onDataReceived(const std::vector<char>& compressedData) {
    double startTime = timeutils::getTimeMicros();

    spdlog::debug("Loading inputData of size {}", compressedData.size());

    // Unpack frame
    const char* ptr = compressedData.data();
    Header header;
    std::memcpy(&header, ptr, sizeof(Header));
    ptr += sizeof(Header);

    // Sanity check
    size_t expectedSize = header.getSize();
    if (compressedData.size() < expectedSize) {
        throw std::runtime_error("Input data size " +
                                  std::to_string(compressedData.size()) +
                                  " is smaller than expected from header " +
                                  std::to_string(expectedSize));
    }

    spdlog::debug("Loading depth size: {}", header.depthSize);

    // Extract pose ID
    pose_id_t poseID = header.poseID;

    // Decompress
    size_t decompressedSize = codec.decompress(ptr, decompressedData, header.depthSize);
    stats.decompressTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
    stats.compressionRatio = static_cast<float>(decompressedSize) / compressedData.size();

    {
        std::unique_lock<std::mutex> lock(m);

        if (frames.size() >= maxQueueSize) {
            FrameData frame = std::move(frames.front());
            frames.pop_front();
            frame.poseID = poseID;
            frame.buffer.resize(decompressedData.size());
            std::memcpy(frame.buffer.data(), decompressedData.data(), frame.buffer.size());
            frames.push_back(std::move(frame));
        }
        else {
            FrameData frame;
            frame.poseID = poseID;
            frame.buffer.resize(decompressedData.size());
            std::memcpy(frame.buffer.data(), decompressedData.data(), frame.buffer.size());
            frames.push_back(std::move(frame));
        }
    }

    stats.receiveTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - prevTime);
    stats.bitrateMbps = ((compressedData.size() * 8) / timeutils::millisToSeconds(stats.receiveTimeMs)) / BYTES_PER_MEGABYTE;

    prevTime = timeutils::getTimeMicros();
}

void BC4DepthVideoTexture::loadFromFile(const Path& dataPath) {
    const std::vector<char>& compressedData = FileIO::loadFromBinaryFile(dataPath);
    onDataReceived(compressedData);
    draw();
}

pose_id_t BC4DepthVideoTexture::draw(pose_id_t poseID) {
    std::lock_guard<std::mutex> lock(m);
    if (frames.empty()) {
        return -1;
    }

    if (poseID != -1 && poseID == prevPoseID) {
        return prevPoseID;
    }

    FrameData* frameData = nullptr;
    if (poseID != -1) {
        for (auto& f : frames) {
            if (f.poseID == poseID) {
                frameData = &f;
                break;
            }
        }
    }
    else {
        frameData = &frames.back();
    }

    if (frameData == nullptr) {
        return -1;
    }

    // Update the BC4 compressed buffer
    bc4CompressedBuffer.bind();
    void* ptr = bc4CompressedBuffer.mapToCPU(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_UNSYNCHRONIZED_BIT);
    if (ptr) {
        std::memcpy(ptr, frameData->buffer.data(), frameData->buffer.size());
        bc4CompressedBuffer.unmapFromCPU();
    }
    else {
        spdlog::warn("Failed to map BC4 compressed buffer. Copying using setData");
        bc4CompressedBuffer.setData(frameData->buffer.size(), frameData->buffer.data());
    }
    bc4CompressedBuffer.unbind();

    prevPoseID = frameData->poseID;

    return prevPoseID;
}
