#include <cstring>
#include <stdexcept>

#include <Utils/FileIO.h>
#include <Utils/TimeUtils.h>
#include <Receivers/QuadsReceiver.h>

using namespace quasar;

QuadsReceiver::QuadsReceiver(QuadSet& quadSet, const std::string& videoURL, const std::string& proxiesURL)
    : quadSet(quadSet)
    , videoURL(videoURL)
    , proxiesURL(proxiesURL)
    , remoteCamera(quadSet.getSize())
    , atlasVideoTexture({
        .width = 2 * quadSet.getSize().x,
        .height = quadSet.getSize().y,
        .internalFormat = GL_RGB,
        .format = GL_RGB,
        .type = GL_UNSIGNED_BYTE,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    }, videoURL)
    , referenceFrameMesh(quadSet, atlasVideoTexture, glm::vec4(0.0f, 0.0f, 0.5f, 1.0f))
    // We can use less vertices and indicies for the mask since it will be sparse
    , residualFrameMesh(quadSet, atlasVideoTexture, glm::vec4(0.5f, 0.0f, 1.0f, 1.0f), MAX_PROXIES_PER_MESH / 4)
    , DataReceiverTCP(proxiesURL)
{
    remoteCameraPrev.setProjectionMatrix(remoteCamera.getProjectionMatrix());
    remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());

    frameInUse = std::make_shared<Frame>(quadSet.getSize());
    framePending = std::make_shared<Frame>(quadSet.getSize());

    threadPool = std::make_unique<BS::thread_pool<>>(4);

    if (!proxiesURL.empty()) {
        spdlog::info("Created QuadsReceiver that recvs from URL: {}", proxiesURL);
    }
}

QuadsReceiver::QuadsReceiver(QuadSet& quadSet, float remoteFOV, const std::string& videoURL, const std::string& proxiesURL)
    : QuadsReceiver(quadSet, videoURL, proxiesURL)
{
    remoteCamera.setFovyDegrees(remoteFOV);
    remoteCameraPrev.setProjectionMatrix(remoteCamera.getProjectionMatrix());
}

void QuadsReceiver::onDataReceived(const std::vector<char>& data) {
    loadFromMemory(data);
}

QuadFrame::FrameType QuadsReceiver::recvData() {
    if (proxiesURL.empty()) {
        return QuadFrame::FrameType::NONE;
    }

    // Wait for a frame that has been written to
    std::shared_ptr<Frame> frame;
    {
        std::unique_lock<std::mutex> lock(m);
        if (!framePending) {
            return QuadFrame::FrameType::NONE;
        }

        pose_id_t videoPoseID = atlasVideoTexture.getLatestPoseID();
        if (videoPoseID < framePending->poseID) { // video is behind, wait until video catches up
            return QuadFrame::FrameType::NONE;
        }

        frame = framePending;
        framePending.reset();
        frameInUse = frame;
    }

    // If video is ahead, draw will search for a previous frame
    atlasVideoTexture.bind();
    atlasVideoTexture.draw(frame->poseID);

    // Reset frame
    QuadFrame::FrameType type = loadFromFrame(frame);
    {
        std::lock_guard<std::mutex> lock(m);
        frameFree = frame;
    }
    cv.notify_one();

    return type;
}

QuadFrame::FrameType QuadsReceiver::loadFromFiles(const Path& dataPath) {
    stats = { 0 };

    double startTime = timeutils::getTimeMicros();

    // Read color data
    Path colorFileName = dataPath / "color.jpg";
    atlasVideoTexture.loadFromFile(colorFileName, true, false);

    // Read previous camera data
    Path cameraFileNamePrev = dataPath / "camera_prev.bin";
    frameInUse->cameraPose.loadFromFile(cameraFileNamePrev);
    frameInUse->cameraPose.copyPoseToCamera(remoteCamera);

    // Read reference frame
    referenceFrame.loadFromFiles(dataPath);
    stats.timeToLoadMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    startTime = timeutils::getTimeMicros();
    frameInUse->frameType = QuadFrame::FrameType::REFERENCE;
    frameInUse->decompressReferenceFrame(threadPool, referenceFrame);
    stats.timeToDecompressMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    // Update reference GPU buffers
    loadFromFrame(frameInUse);

    startTime = timeutils::getTimeMicros();

    // Read camera data
    Path cameraFileName = dataPath / "camera.bin";
    frameInUse->cameraPose.loadFromFile(cameraFileName);
    frameInUse->cameraPose.copyPoseToCamera(remoteCamera);

    // Read residual frame
    residualFrame.loadFromFiles(dataPath);
    stats.timeToLoadMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    startTime = timeutils::getTimeMicros();
    frameInUse->frameType = QuadFrame::FrameType::RESIDUAL;
    frameInUse->decompressResidualFrame(threadPool, residualFrame);
    stats.timeToDecompressMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    // Update residual GPU buffers
    loadFromFrame(frameInUse);

    return frameInUse->frameType;
}

QuadFrame::FrameType QuadsReceiver::loadFromMemory(const std::vector<char>& inputData) {
    stats = { 0 };

    double startTime = timeutils::getTimeMicros();

    spdlog::debug("Loading inputData of size {}", inputData.size());

    // Unpack frame
    const char* ptr = inputData.data();
    Header header;
    std::memcpy(&header, ptr, sizeof(Header));
    ptr += sizeof(Header);

    // Sanity check
    size_t expectedSize = sizeof(Header) +
                          header.cameraSize +
                          header.geometrySize;
    if (inputData.size() < expectedSize) {
        throw std::runtime_error("Input data size " +
                                  std::to_string(inputData.size()) +
                                  " is smaller than expected from header " +
                                  std::to_string(expectedSize));
    }

    // Wait for a free frame
    std::shared_ptr<Frame> frame;
    {
        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock, [&]() { return frameFree != nullptr; });
        frame = frameFree;
        frameFree.reset();
    }
    frame->poseID = header.poseID;
    frame->frameType = header.frameType;

    spdlog::debug("Loading camera size: {}", header.cameraSize);
    spdlog::debug("Loading geometry size: {}", header.geometrySize);

    // Read camera data
    frame->cameraPose.loadFromMemory(ptr, header.cameraSize);
    ptr += header.cameraSize;

    // Read geometry data
    geometryData.resize(header.geometrySize);
    std::memcpy(geometryData.data(), ptr, header.geometrySize);
    ptr += header.geometrySize;

    stats.timeToLoadMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    if (header.frameType == QuadFrame::FrameType::REFERENCE) {
        startTime = timeutils::getTimeMicros();
        referenceFrame.loadFromMemory(geometryData.data(), header.geometrySize);
        stats.timeToLoadMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        startTime = timeutils::getTimeMicros();
        frame->decompressReferenceFrame(threadPool, referenceFrame);
        stats.timeToDecompressMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
    }
    else {
        startTime = timeutils::getTimeMicros();
        residualFrame.loadFromMemory(geometryData.data(), header.geometrySize);
        stats.timeToLoadMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        startTime = timeutils::getTimeMicros();
        frame->decompressResidualFrame(threadPool, residualFrame);
        stats.timeToDecompressMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
    }

    // Signal that frame is ready
    {
        std::lock_guard<std::mutex> lock(m);
        framePending = frame;
    }
    cv.notify_one();

    return frame->frameType;
}

QuadFrame::FrameType QuadsReceiver::loadFromFrame(std::shared_ptr<Frame> frame) {
    if (frame->frameType == QuadFrame::FrameType::NONE) {
        return QuadFrame::FrameType::NONE;
    }

    spdlog::debug("Reconstructing {} Frame...", frame->frameType == QuadFrame::FrameType::REFERENCE ? "Reference" : "Residual");
    frame->cameraPose.copyPoseToCamera(remoteCamera);

    const glm::vec2& gBufferSize = quadSet.getSize();
    double startTime = timeutils::getTimeMicros();
    if (frame->frameType == QuadFrame::FrameType::REFERENCE) {
        // Transfer proxies to GPU for reconstruction
        auto sizes = quadSet.loadFromMemory(frame->uncompressedQuads, frame->uncompressedOffsets);
        referenceFrame.numQuads = sizes.numQuads;
        referenceFrame.numDepthOffsets = sizes.numDepthOffsets;
        stats.timeToTransferMs = quadSet.stats.timeToTransferMs;

        // Using GPU buffers, reconstruct mesh using proxies
        startTime = timeutils::getTimeMicros();
        referenceFrameMesh.appendQuads(quadSet, gBufferSize);
        referenceFrameMesh.createMeshFromProxies(quadSet, gBufferSize, remoteCamera);
        stats.timeToCreateMeshMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        auto refMeshBufferSizes = referenceFrameMesh.getBufferSizes();
        stats.totalTriangles = refMeshBufferSizes.numIndices / 3;
        stats.sizes = sizes;

        remoteCameraPrev.setProjectionMatrix(remoteCamera.getProjectionMatrix());
        remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());
    }
    else {
        // Transfer updated proxies to GPU for reconstruction
        auto sizesUpdated = quadSet.loadFromMemory(frame->uncompressedQuads, frame->uncompressedOffsets);
        residualFrame.numQuadsUpdated = sizesUpdated.numQuads;
        residualFrame.numDepthOffsetsUpdated = sizesUpdated.numDepthOffsets;
        stats.timeToTransferMs = quadSet.stats.timeToTransferMs;

        // Using GPU buffers, update reference frame mesh using proxies
        startTime = timeutils::getTimeMicros();
        referenceFrameMesh.appendQuads(quadSet, gBufferSize, false /* not a reference frame */);
        referenceFrameMesh.createMeshFromProxies(quadSet, gBufferSize, remoteCameraPrev);
        stats.timeToCreateMeshMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        // This will also wait for the GPU to finish
        auto refMeshBufferSizes = referenceFrameMesh.getBufferSizes();
        stats.totalTriangles = refMeshBufferSizes.numIndices / 3;

        // Transfer revealed proxies to GPU for reconstruction
        auto sizesRevealed = quadSet.loadFromMemory(frame->uncompressedQuadsRevealed, frame->uncompressedOffsetsRevealed);
        residualFrame.numQuadsRevealed = sizesRevealed.numQuads;
        residualFrame.numDepthOffsetsRevealed = sizesRevealed.numDepthOffsets;
        stats.timeToTransferMs += quadSet.stats.timeToTransferMs;

        // Using GPU buffers, reconstruct revealed mesh using proxies
        startTime = timeutils::getTimeMicros();
        residualFrameMesh.appendQuads(quadSet, gBufferSize);
        residualFrameMesh.createMeshFromProxies(quadSet, gBufferSize, remoteCamera);
        stats.timeToCreateMeshMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        auto resMeshBufferSizes = residualFrameMesh.getBufferSizes();
        stats.totalTriangles += resMeshBufferSizes.numIndices / 3;
        stats.sizes += sizesUpdated + sizesRevealed;
    }

    return frame->frameType;
}
