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
    , videoAtlasTexture({
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
    , alphaAtlasTexture({
        .width = 2 * quadSet.getSize().x,
        .height = quadSet.getSize().y,
        .internalFormat = GL_R8,
        .format = GL_RED,
        .type = GL_UNSIGNED_BYTE,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE
    })
    , alphaCodec(alphaAtlasTexture.width, alphaAtlasTexture.height)
    , referenceFrameMesh(quadSet, videoAtlasTexture, alphaAtlasTexture, glm::vec4(0.0f, 0.0f, 0.5f, 1.0f))
    // We can use less vertices and indicies for the mask since it will be sparse
    , residualFrameMesh(quadSet, videoAtlasTexture, alphaAtlasTexture, glm::vec4(0.5f, 0.0f, 1.0f, 1.0f))
    , DataReceiverTCP(proxiesURL)
{
    remoteCameraPrev.setProjectionMatrix(remoteCamera.getProjectionMatrix());
    remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());

    frameInUse = std::make_shared<Frame>(quadSet.getSize());
    framePending = std::make_shared<Frame>(quadSet.getSize());

    frameFree = framePending;
    cv.notify_one();

    threadPool = std::make_unique<BS::thread_pool<>>(4);

    if (!proxiesURL.empty()) {
        spdlog::info("Created QuadsReceiver that recvs from URL: tcp://{}", proxiesURL);
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
    QuadFrame::FrameType frameType = QuadFrame::FrameType::NONE;

    if (proxiesURL.empty()) {
        return frameType;
    }

    if (!videoAtlasTexture.containsFrames()) {
        return frameType;
    }

    // Wait for a frame that has been written to
    std::shared_ptr<Frame> frame;
    {
        std::unique_lock<std::mutex> lock(m);
        if (!framePending) {
            return frameType;
        }

        if (videoAtlasTexture.getLatestPoseID() < framePending->poseID) { // Video is behind, wait until video catches up
            return frameType;
        }

        frame = framePending;
        framePending.reset();
        frameInUse = frame;
    }

    // If video is ahead, search for a previous frame
    if (!videoAtlasTexture.containsFrameWithPoseID(frame->poseID)) {
        // This means we dropped a video frame. We have to wait for the next reference frame to resync
        waitUntilReferenceFrame = true;
    }
    else if (!waitUntilReferenceFrame || (waitUntilReferenceFrame && frame->frameType == QuadFrame::FrameType::REFERENCE)) {
        // Update color texture
        videoAtlasTexture.bind();
        videoAtlasTexture.draw(frame->poseID);

        // Update alpha texture
        alphaAtlasTexture.bind();
        alphaAtlasTexture.loadFromData(frame->alphaData.data());

        // Reconstruct meshes from frame
        frameType = reconstructFrame(frame);

        // Video and proxies are synced now, no need to wait for reference frame anymore
        waitUntilReferenceFrame = false;
    }

    // Reset frame
    {
        std::lock_guard<std::mutex> lock(m);
        frameFree = frame;
    }
    cv.notify_one();

    return frameType;
}

QuadFrame::FrameType QuadsReceiver::loadFromFiles(const Path& dataPath) {
    stats = { 0 };

    double startTime = timeutils::getTimeMicros();

    // Read color data
    Path colorFileName = dataPath / "color.jpg";
    videoAtlasTexture.loadFromFile(colorFileName, true, false);

    // Read alpha data
    Path alphaFileName = dataPath / "alpha.png";
    alphaAtlasTexture.loadFromFile(alphaFileName, true, false);

    // Read previous camera data
    Path cameraFileNamePrev = dataPath / "camera_prev.bin";
    frameInUse->cameraPose.loadFromFile(cameraFileNamePrev);
    frameInUse->cameraPose.copyPoseToCamera(remoteCamera);

    // Read reference frame
    referenceFrame.loadFromFiles(dataPath);
    stats.loadTimeMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    startTime = timeutils::getTimeMicros();
    frameInUse->frameType = QuadFrame::FrameType::REFERENCE;
    size_t refSize = frameInUse->decompressReferenceFrame(threadPool, referenceFrame);
    stats.decompressTimeMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    // Update reference GPU buffers
    if (refSize > 0) reconstructFrame(frameInUse);

    startTime = timeutils::getTimeMicros();

    // Read camera data
    Path cameraFileName = dataPath / "camera.bin";
    frameInUse->cameraPose.loadFromFile(cameraFileName);
    frameInUse->cameraPose.copyPoseToCamera(remoteCamera);

    // Read residual frame
    residualFrame.loadFromFiles(dataPath);
    stats.loadTimeMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    startTime = timeutils::getTimeMicros();
    frameInUse->frameType = QuadFrame::FrameType::RESIDUAL;
    size_t resSize = frameInUse->decompressResidualFrame(threadPool, residualFrame);
    stats.decompressTimeMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    // Update residual GPU buffers
    if (resSize > 0) reconstructFrame(frameInUse);

    return frameInUse->frameType;
}

QuadFrame::FrameType QuadsReceiver::loadFromMemory(const std::vector<char>& inputData) {
    double startTime = timeutils::getTimeMicros();

    spdlog::debug("Loading inputData of size {}", inputData.size());

    // Unpack frame
    const char* ptr = inputData.data();
    Header header;
    std::memcpy(&header, ptr, sizeof(Header));
    ptr += sizeof(Header);

    // Sanity check
    size_t expectedSize = header.getSize();
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
    spdlog::debug("Loading alpha size: {}", header.alphaSize);
    spdlog::debug("Loading geometry size: {}", header.geometrySize);

    // Read camera data
    frame->cameraPose.loadFromMemory(ptr, header.cameraSize);
    ptr += header.cameraSize;

    // Read alpha data
    alphaCodec.decompress(ptr, frame->alphaData, header.alphaSize);
    ptr += header.alphaSize;

    // Read geometry data
    if (header.frameType == QuadFrame::FrameType::REFERENCE) {
        startTime = timeutils::getTimeMicros();
        referenceFrame.loadFromMemory(ptr, header.geometrySize);
    }
    else {
        startTime = timeutils::getTimeMicros();
        residualFrame.loadFromMemory(ptr, header.geometrySize);
    }
    ptr += header.geometrySize;
    stats.loadTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    // Decompress (asynchronous)
    startTime = timeutils::getTimeMicros();
    if (header.frameType == QuadFrame::FrameType::REFERENCE) {
        frame->decompressReferenceFrame(threadPool, referenceFrame);
    }
    else {
        frame->decompressResidualFrame(threadPool, residualFrame);
    }
    stats.decompressTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    // Signal that frame is ready
    {
        std::lock_guard<std::mutex> lock(m);
        framePending = frame;
    }
    cv.notify_one();

    return frame->frameType;
}

QuadFrame::FrameType QuadsReceiver::reconstructFrame(std::shared_ptr<Frame> frame) {
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
        stats.transferTimeMs = quadSet.stats.transferTimeMs;

        // Using GPU buffers, reconstruct mesh using proxies
        startTime = timeutils::getTimeMicros();
        referenceFrameMesh.appendQuads(quadSet, gBufferSize);
        referenceFrameMesh.createMeshFromProxies(quadSet, gBufferSize, remoteCamera);
        stats.createMeshTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

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
        stats.transferTimeMs = quadSet.stats.transferTimeMs;

        // Using GPU buffers, update reference frame mesh using proxies
        startTime = timeutils::getTimeMicros();
        referenceFrameMesh.appendQuads(quadSet, gBufferSize, false /* not a reference frame */);
        referenceFrameMesh.createMeshFromProxies(quadSet, gBufferSize, remoteCameraPrev);
        stats.createMeshTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        // This will also wait for the GPU to finish
        auto refMeshBufferSizes = referenceFrameMesh.getBufferSizes();
        stats.totalTriangles = refMeshBufferSizes.numIndices / 3;

        // Transfer revealed proxies to GPU for reconstruction
        auto sizesRevealed = quadSet.loadFromMemory(frame->uncompressedQuadsRevealed, frame->uncompressedOffsetsRevealed);
        residualFrame.numQuadsRevealed = sizesRevealed.numQuads;
        residualFrame.numDepthOffsetsRevealed = sizesRevealed.numDepthOffsets;
        stats.transferTimeMs += quadSet.stats.transferTimeMs;

        // Using GPU buffers, reconstruct revealed mesh using proxies
        startTime = timeutils::getTimeMicros();
        residualFrameMesh.appendQuads(quadSet, gBufferSize);
        residualFrameMesh.createMeshFromProxies(quadSet, gBufferSize, remoteCamera);
        stats.createMeshTimeMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        auto resMeshBufferSizes = residualFrameMesh.getBufferSizes();
        stats.totalTriangles += resMeshBufferSizes.numIndices / 3;
        stats.sizes = sizesUpdated + sizesRevealed;
    }

    return frame->frameType;
}
