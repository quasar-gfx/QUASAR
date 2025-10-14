#include <spdlog/spdlog.h>

#include <Streamers/BC4DepthStreamer.h>
#include <shaders_common.h>

#ifndef __ANDROID__
#define THREADS_PER_LOCALGROUP 32
#else
#define THREADS_PER_LOCALGROUP 16
#endif

using namespace quasar;

BC4DepthStreamer::BC4DepthStreamer(const RenderTargetCreateParams& params, const std::string& receiverURL, uint maxFrameRate)
    : receiverURL(receiverURL)
    , maxFrameRate(maxFrameRate)
    , width((params.width + BC4_BLOCK_SIZE - 1) / BC4_BLOCK_SIZE * BC4_BLOCK_SIZE)
    , height((params.height + BC4_BLOCK_SIZE - 1) / BC4_BLOCK_SIZE * BC4_BLOCK_SIZE)
    , compressedSize((width / BC4_BLOCK_SIZE) * (height / BC4_BLOCK_SIZE))
    , dataBC4(compressedSize * sizeof(BC4Block))
    , bc4CompressionShader({
        .computeCodeData = SHADER_COMMON_BC4_COMPRESS_COMP,
        .computeCodeSize = SHADER_COMMON_BC4_COMPRESS_COMP_len,
        .defines = {
            "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
        }
    })
    , RenderTarget(params)
    , DataStreamerTCP(receiverURL)
{
    resize(width, height);

    bc4CompressedBuffer = Buffer({
        .target = GL_SHADER_STORAGE_BUFFER,
        .dataSize = sizeof(BC4Block),
        .numElems = compressedSize,
        .usage = GL_DYNAMIC_DRAW,
    });

    if (!receiverURL.empty()) {
        spdlog::info("Created BC4DepthStreamer that sends to URL: tcp://{}", receiverURL);
    }
}

size_t BC4DepthStreamer::generateFrame() {
    glm::vec2 depthMapSize{width, height};

    // Compress with BC4
    bc4CompressionShader.bind();
    {
        bc4CompressionShader.setTexture(colorTexture, 0);
    }
    {
        bc4CompressionShader.setVec2("depthMapSize", depthMapSize);
        bc4CompressionShader.setVec2("bc4DepthSize", depthMapSize / static_cast<float>(BC4_BLOCK_SIZE));
        bc4CompressionShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, bc4CompressedBuffer);
    }
    bc4CompressionShader.dispatch(((depthMapSize.x / BC4_BLOCK_SIZE) + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                  ((depthMapSize.y / BC4_BLOCK_SIZE) + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
    bc4CompressionShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    double startTime = timeutils::getTimeMicros();

    // Copy depth data
    bc4CompressedBuffer.bind();
    void* ptr = bc4CompressedBuffer.mapToCPU(GL_MAP_READ_BIT);
    if (ptr) {
        std::memcpy(dataBC4.data(), ptr, compressedSize * sizeof(BC4Block));
        bc4CompressedBuffer.unmapFromCPU();
    }
    else {
        spdlog::error("Failed to map BC4 compressed buffer to CPU. Copying using getData");
        bc4CompressedBuffer.getData(dataBC4.data());
    }
    stats.transferTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    // Compress with codec
    startTime = timeutils::getTimeMicros();
    size_t compressedSize = codec.compress(dataBC4.data(), dataZSTD, dataBC4.size());
    dataZSTD.resize(compressedSize);
    stats.compressTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
    stats.compressionRatio = compressedSize / static_cast<float>(dataBC4.size());

    return compressedSize;
}

size_t BC4DepthStreamer::writeToFile(const Path& filename) {
    writeToMemory(-1, compressedData);
    FileIO::writeToBinaryFile(filename.str(), compressedData.data(), compressedData.size());
    return compressedData.size();
}

size_t BC4DepthStreamer::writeToMemory(pose_id_t poseID, std::vector<char>& outputData) {
    BC4DepthVideoTexture::Header header{
        .poseID = poseID,
        .depthSize = static_cast<uint32_t>(dataZSTD.size())
    };

    spdlog::debug("Writing depth size: {}", header.depthSize);

    outputData.resize(header.getSize());
    char* ptr = outputData.data();

    // Write header
    std::memcpy(ptr, &header, sizeof(header));
    ptr += sizeof(header);

    // Write compressed data
    std::memcpy(ptr, dataZSTD.data(), dataZSTD.size());
    ptr += dataZSTD.size();

    return outputData.size();
}

void BC4DepthStreamer::sendFrame(pose_id_t poseID) {
    size_t outputSize = writeToMemory(poseID, compressedData);

    send(compressedData);

    stats.sendTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - prevTime);
    stats.bitrateMbps = ((8.0 * outputSize) / BYTES_PER_MEGABYTE) / timeutils::millisToSeconds(stats.sendTimeMs);

    prevTime = timeutils::getTimeMicros();
}
