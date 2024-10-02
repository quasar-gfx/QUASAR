#include "BC4DepthVideoTexture.h"
#include <GL/glew.h>
#include <Utils/TimeUtils.h>

BC4DepthVideoTexture::BC4DepthVideoTexture(const TextureDataCreateParams &params, std::string streamerURL)
    : streamerURL(streamerURL)
    , DataReceiverTCP(streamerURL, false)
    , Texture(params) {

    compressedSize = (params.width / 8) * (params.height / 8) * sizeof(Block);

    // Load and compile BC4 decompression compute shader
    bc4DecompressComputeShader = loadComputeShader("shaders/bc4Decompression.comp");

    // Create BC4 compressed buffer
    glGenBuffers(1, &bc4CompressedBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, bc4CompressedBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, compressedSize, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

BC4DepthVideoTexture::~BC4DepthVideoTexture() {
    glDeleteBuffers(1, &bc4CompressedBuffer);
    glDeleteProgram(bc4DecompressComputeShader);
}

pose_id_t BC4DepthVideoTexture::getLatestPoseID() {
    std::lock_guard<std::mutex> lock(m);
    return latestPoseID;
}

void BC4DepthVideoTexture::onDataReceived(const std::vector<uint8_t>& data) {
    std::lock_guard<std::mutex> lock(m);
    
    float startTime = timeutils::getTimeMicros();

    pose_id_t poseID;
    std::memcpy(&poseID, data.data(), sizeof(pose_id_t));

    std::vector<uint8_t> compressedData(data.begin() + sizeof(pose_id_t), data.end());

    decompressBC4(compressedData);

    latestPoseID = poseID;

    stats.timeToReceiveMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
    stats.bitrateMbps = ((sizeof(pose_id_t) + compressedData.size() * 8) / timeutils::millisToSeconds(stats.timeToReceiveMs)) / MBPS_TO_BPS;
}

pose_id_t BC4DepthVideoTexture::draw(pose_id_t poseID) {
    // Since we decompress immediately on receive, draw just needs to ensure the texture is bound
    glBindTexture(GL_TEXTURE_2D, ID);
    return latestPoseID;
}

void BC4DepthVideoTexture::decompressBC4(const std::vector<uint8_t>& compressedData) {
    float startTime = timeutils::getTimeMicros();

    glUseProgram(bc4DecompressComputeShader);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bc4CompressedBuffer);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, compressedSize, compressedData.data());

    glBindImageTexture(0, ID, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R16F);

    glUniform2f(glGetUniformLocation(bc4DecompressComputeShader, "depthMapSize"), width, height);

    glDispatchCompute((width + 15) / 16, (height + 15) / 16, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    stats.timeToDecompressMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
}