#include <cstring>
#include <stdexcept>

#include <Utils/FileIO.h>
#include <Utils/TimeUtils.h>
#include <Receivers/QUASARReceiver.h>

using namespace quasar;

QUASARReceiver::QUASARReceiver(QuadSet& quadSet, uint maxLayers, const std::string& videoURL, const std::string& proxiesURL)
    : quadSet(quadSet)
    , maxLayers(maxLayers)
    , videoURL(videoURL)
    , proxiesURL(proxiesURL)
    , remoteCamera(quadSet.getSize())
    , remoteCameraWideFOV(quadSet.getSize())
    , atlasVideoTexture({
        .width = 2 * quadSet.getSize().x,
        .height = 3 * quadSet.getSize().y,
        .internalFormat = GL_RGB,
        .format = GL_RGB,
        .type = GL_UNSIGNED_BYTE,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    }, videoURL)
    // We can use less vertices and indicies for the mask since it will be sparse
    , residualFrameMesh(quadSet, atlasVideoTexture, glm::vec4(0.5f, 2.0f / 3.0f, 1.0f, 1.0f), MAX_PROXIES_PER_MESH / 4)
    , DataReceiverTCP(proxiesURL)
{
    meshes.reserve(maxLayers);
    referenceFrames.resize(maxLayers);

    remoteCameraPrev.setProjectionMatrix(remoteCamera.getProjectionMatrix());
    remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());

    // Untile texture atlas
    glm::vec4 textureExtent(0.0f);
    for (int layer = 0; layer < maxLayers; layer++) {
        // First and last layer need a lot of quads, each subsequent one has less
        uint maxProxies =
            (layer == 0 || layer == maxLayers - 1) ? MAX_PROXIES_PER_MESH :
                (layer == 1) ? MAX_PROXIES_PER_MESH / 4 : MAX_PROXIES_PER_MESH / 8;
        textureExtent.z = textureExtent.x + 0.5f;
        textureExtent.w = textureExtent.y + 1.0f / 3.0f;
        meshes.emplace_back(quadSet, atlasVideoTexture, textureExtent, maxProxies);

        textureExtent.x += 0.5f;
        if (textureExtent.x >= 1.0f) {
            textureExtent.x = 0.0f;
            textureExtent.y += 1.0f / 3.0f;
        }
    }

    frameInUse = std::make_shared<Frame>(quadSet.getSize(), maxLayers);
    framePending = std::make_shared<Frame>(quadSet.getSize(), maxLayers);

    threadPool = std::make_unique<BS::thread_pool<>>(6);

    if (!proxiesURL.empty()) {
        spdlog::info("Created QUASARReceiver that recvs from URL: {}", proxiesURL);
    }
}

QUASARReceiver::QUASARReceiver(
        QuadSet& quadSet,
        uint maxLayers, float remoteFOV, float remoteFOVWide,
        const std::string& videoURL, const std::string& proxiesURL)
    : QUASARReceiver(quadSet, maxLayers, videoURL, proxiesURL)
{
    remoteCamera.setFovyDegrees(remoteFOV);
    remoteCameraPrev.setProjectionMatrix(remoteCamera.getProjectionMatrix());

    remoteCameraWideFOV.setFovyDegrees(remoteFOVWide);
    remoteCameraWideFOV.setViewMatrix(remoteCamera.getViewMatrix());
}

void QUASARReceiver::onDataReceived(const std::vector<char>& data) {
    loadFromMemory(data);
}

QuadFrame::FrameType QUASARReceiver::recvData() {
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

QuadFrame::FrameType QUASARReceiver::loadFromFiles(const Path& dataPath) {
    stats = { 0 };

    double startTime = timeutils::getTimeMicros();

    // Read color data
    Path colorFileName = dataPath / "color.jpg";
    atlasVideoTexture.loadFromFile(colorFileName, true, false);

    // Read previous camera data
    Path cameraFileNamePrev = dataPath / "camera_prev.bin";
    frameInUse->cameraPose.loadFromFile(cameraFileNamePrev);
    frameInUse->cameraPose.copyPoseToCamera(remoteCamera);
    frameInUse->cameraPose.copyPoseToCamera(remoteCameraWideFOV, false);

    // Read metadata (viewSphereDiameter and wide FOV)
    const auto& metadata = FileIO::loadFromBinaryFile(dataPath / "metadata.bin");
    Params params;
    std::memcpy(&params, metadata.data(), metadata.size());

    if (params.numLayers != maxLayers) {
        spdlog::warn("Loaded number of layers {} does not match initialized number of layers {}", params.numLayers, maxLayers);
        maxLayers = params.numLayers;
    }
    spdlog::debug("Loaded wide FOV: {}", params.wideFOV);
    spdlog::debug("Loaded view sphere diameter: {}", params.viewSphereDiameter);

    remoteCameraWideFOV.setFovyDegrees(params.wideFOV);
    setViewSphereDiameter(params.viewSphereDiameter);

    // Read reference frames
    for (int layer = 0; layer < maxLayers ; layer++) {
        referenceFrames[layer].loadFromFiles(dataPath, layer);
    }
    stats.timeToLoadMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    startTime = timeutils::getTimeMicros();
    frameInUse->frameType = QuadFrame::FrameType::REFERENCE;
    frameInUse->decompressReferenceFrames(threadPool, referenceFrames);
    stats.timeToDecompressMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    // Update reference GPU buffers
    loadFromFrame(frameInUse);

    startTime = timeutils::getTimeMicros();

    // Read camera data
    Path cameraFileName = dataPath / "camera.bin";
    frameInUse->cameraPose.loadFromFile(cameraFileName);
    frameInUse->cameraPose.copyPoseToCamera(remoteCamera);
    frameInUse->cameraPose.copyPoseToCamera(remoteCameraWideFOV, false);

    // Read residual frame
    residualFrame.loadFromFiles(dataPath);
    stats.timeToLoadMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    startTime = timeutils::getTimeMicros();
    frameInUse->frameType = QuadFrame::FrameType::RESIDUAL;
    frameInUse->decompressReferenceAndResidualFrames(threadPool, referenceFrames, residualFrame);
    stats.timeToDecompressMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    // Update residual GPU buffers
    loadFromFrame(frameInUse);

    return frameInUse->frameType;
}

QuadFrame::FrameType QUASARReceiver::loadFromMemory(const std::vector<char>& inputData) {
    stats = { 0 };

    double startTime = timeutils::getTimeMicros();

    spdlog::debug("Loading inputData of size {}", inputData.size());

    const char* ptr = inputData.data();

    // Read header
    Header header;
    std::memcpy(&header, ptr, sizeof(Header));
    ptr += sizeof(Header);

    size_t expectedSize = sizeof(Header) +
                          header.cameraSize +
                          header.geometrySize;
    if (inputData.size() < expectedSize) {
        throw std::runtime_error("Input data size " +
                                 std::to_string(inputData.size()) +
                                 " is smaller than expected from header " +
                                 std::to_string(expectedSize));
    }

    std::shared_ptr<Frame> frame;
    {
        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock, [&]() { return frameFree != nullptr; });
        frame = frameFree;
        frameFree.reset();
    }

    frame->poseID = header.poseID;
    frame->frameType = header.frameType;

    maxLayers = header.params.numLayers;
    setViewSphereDiameter(header.params.viewSphereDiameter);
    remoteCameraWideFOV.setFovyDegrees(header.params.wideFOV);

    spdlog::debug("Loading camera size: {}", header.cameraSize);
    spdlog::debug("Loading geometry size: {}", header.geometrySize);

    // Read camera data
    frame->cameraPose.loadFromMemory(ptr, header.cameraSize);
    ptr += header.cameraSize;

    // Read geometry data
    geometryData.resize(header.geometrySize);
    std::memcpy(geometryData.data(), ptr, header.geometrySize);

    stats.timeToLoadMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    // Read visible layer
    uint32_t layerSize = 0;
    if (header.frameType == QuadFrame::FrameType::REFERENCE) {
        startTime = timeutils::getTimeMicros();

        // Read size of layer
        std::memcpy(&layerSize, ptr, sizeof(uint32_t));
        ptr += sizeof(uint32_t);

        // Read layer data
        referenceFrames[0].loadFromMemory(ptr, layerSize);
        ptr += layerSize;

        stats.timeToLoadMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
    }
    else {
        startTime = timeutils::getTimeMicros();

        // Read size of layer
        std::memcpy(&layerSize, ptr, sizeof(uint32_t));
        ptr += sizeof(uint32_t);

        // Read layer data
        residualFrame.loadFromMemory(ptr, layerSize);
        ptr += layerSize;

        stats.timeToLoadMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
    }

    // Read hidden layers and wide FOV
    for (int layer = 1; layer < maxLayers; layer++) {
        // Read size of layer
        std::memcpy(&layerSize, ptr, sizeof(uint32_t));
        ptr += sizeof(uint32_t);

        // Read layer data
        referenceFrames[layer].loadFromMemory(ptr, layerSize);
        ptr += layerSize;
    }

    if (header.frameType == QuadFrame::FrameType::REFERENCE) {
        startTime = timeutils::getTimeMicros();
        frame->decompressReferenceFrames(threadPool, referenceFrames);
        stats.timeToDecompressMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
    }
    else {
       startTime = timeutils::getTimeMicros();
       frame->decompressReferenceAndResidualFrames(threadPool, referenceFrames, residualFrame);
       stats.timeToDecompressMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
    }

    // Signal that frame is ready
    {
        std::lock_guard<std::mutex> lock(m);
        framePending = frame;
    }
    cv.notify_one();

    return frame->frameType;
}

QuadFrame::FrameType QUASARReceiver::loadFromFrame(std::shared_ptr<Frame> frame) {
    if (frame->frameType == QuadFrame::FrameType::NONE) {
        return QuadFrame::FrameType::NONE;
    }

    spdlog::debug("Reconstructing {} Frame...", frame->frameType == QuadFrame::FrameType::REFERENCE ? "Reference" : "Residual");
    frame->cameraPose.copyPoseToCamera(remoteCamera);
    frame->cameraPose.copyPoseToCamera(remoteCameraWideFOV, false);

    const glm::vec2& gBufferSize = quadSet.getSize();
    double startTime = timeutils::getTimeMicros();
    if (frame->frameType == QuadFrame::FrameType::REFERENCE) {
        // Transfer proxies to GPU for reconstruction
        auto sizes = quadSet.loadFromMemory(frame->uncompressedQuads[0], frame->uncompressedOffsets[0]);
        referenceFrames[0].numQuads = sizes.numQuads;
        referenceFrames[0].numDepthOffsets = sizes.numDepthOffsets;
        stats.timeToTransferMs += quadSet.stats.timeToTransferMs;

        // Using GPU buffers, reconstruct mesh using proxies
        auto& cameraToUse = getCameraToUse(0);
        startTime = timeutils::getTimeMicros();
        meshes[0].appendQuads(quadSet, gBufferSize);
        meshes[0].createMeshFromProxies(quadSet, gBufferSize, cameraToUse);
        stats.timeToCreateMeshMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        auto meshBufferSizes = meshes[0].getBufferSizes();
        stats.totalTriangles += meshBufferSizes.numIndices / 3;
        stats.sizes += sizes;

        remoteCameraPrev.setProjectionMatrix(remoteCamera.getProjectionMatrix());
        remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());
    }
    else {
        // Transfer updated proxies to GPU for reconstruction
        auto sizesUpdated = quadSet.loadFromMemory(frame->uncompressedQuads[0], frame->uncompressedOffsets[0]);
        residualFrame.numQuadsUpdated = sizesUpdated.numQuads;
        residualFrame.numDepthOffsetsUpdated = sizesUpdated.numDepthOffsets;
        stats.timeToTransferMs = quadSet.stats.timeToTransferMs;

        // Using GPU buffers, update reference frame mesh using proxies
        startTime = timeutils::getTimeMicros();
        meshes[0].appendQuads(quadSet, gBufferSize, false /* not a reference frame */);
        meshes[0].createMeshFromProxies(quadSet, gBufferSize, remoteCameraPrev);
        stats.timeToCreateMeshMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        // This will also wait for the GPU to finish
        auto refMeshBufferSizes = meshes[0].getBufferSizes();
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

    // Reconstruct hidden layers and wide FOV
    for (int layer = 1; layer < maxLayers; ++layer) {
        // Transfer proxies to GPU for reconstruction
        auto sizes = quadSet.loadFromMemory(frame->uncompressedQuads[layer], frame->uncompressedOffsets[layer]);
        referenceFrames[layer].numQuads = sizes.numQuads;
        referenceFrames[layer].numDepthOffsets = sizes.numDepthOffsets;
        stats.timeToTransferMs += quadSet.stats.timeToTransferMs;

        // Using GPU buffers, reconstruct mesh using proxies
        const auto& cameraToUse = getCameraToUse(layer);
        startTime = timeutils::getTimeMicros();
        meshes[layer].appendQuads(quadSet, gBufferSize);
        meshes[layer].createMeshFromProxies(quadSet, gBufferSize, cameraToUse);
        stats.timeToCreateMeshMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        auto meshBufferSizes = meshes[layer].getBufferSizes();
        stats.totalTriangles += meshBufferSizes.numIndices / 3;
        stats.sizes += sizes;
    }

    return frame->frameType;
}
