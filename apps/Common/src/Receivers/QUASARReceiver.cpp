#include <Receivers/QUASARReceiver.h>

using namespace quasar;

QUASARReceiver::QUASARReceiver(QuadSet& quadSet, uint maxLayers, const std::string& videoURL, const std::string& proxiesURL)
    : quadSet(quadSet)
    , maxLayers(maxLayers)
    , remoteCamera(quadSet.getSize())
    , remoteCameraWideFOV(quadSet.getSize())
    , frames(maxLayers)
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
    , uncompressedQuads(sizeof(uint) + quadSet.quadBuffers.maxProxies * sizeof(QuadMapDataPacked))
    , uncompressedOffsets(quadSet.depthOffsets.getSize().x * quadSet.depthOffsets.getSize().y * 4 * sizeof(uint16_t))
    , DataReceiverTCP(proxiesURL)
{
    meshes.reserve(maxLayers);

    remoteCameraPrev.setProjectionMatrix(remoteCamera.getProjectionMatrix());
    remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());

    // Untile texture atlas
    glm::vec4 textureExtent(0.0f);
    for (int layer = 0; layer < maxLayers; layer++) {
        textureExtent.z = textureExtent.x + 0.5f;
        textureExtent.w = textureExtent.y + 1.0f / 3.0f;

        // First and last layer need a lot of quads, each subsequent one has less
        uint maxProxies = (layer == 0 || layer == maxLayers - 1) ? MAX_QUADS_PER_MESH : MAX_QUADS_PER_MESH / 4;
        meshes.emplace_back(quadSet, atlasVideoTexture, textureExtent, maxProxies);

        textureExtent.x += 0.5f;
        if (textureExtent.x >= 1.0f) {
            textureExtent.x = 0.0f;
            textureExtent.y += 1.0f / 3.0f;
            if (textureExtent.y >= 1.0f) textureExtent.y = 0.0f; // optional safety wrap
        }
    }

    threadPool = std::make_unique<BS::thread_pool<>>(2);
}

QUASARReceiver::QUASARReceiver(QuadSet& quadSet, uint maxLayers, float remoteFOV, float remoteFOVWide, const std::string& videoURL, const std::string& proxiesURL)
    : QUASARReceiver(quadSet, maxLayers, videoURL, proxiesURL)
{
    remoteCamera.setFovyDegrees(remoteFOV);
    remoteCameraPrev.setProjectionMatrix(remoteCamera.getProjectionMatrix());

    remoteCameraWideFOV.setFovyDegrees(remoteFOVWide);
    remoteCameraWideFOV.setViewMatrix(remoteCamera.getViewMatrix());
}

void QUASARReceiver::updateViewSphere(float viewSphereDiameter) {
    this->viewSphereDiameter = viewSphereDiameter;
}

void QUASARReceiver::onDataReceived(const std::vector<char>& data) {
    // loadFromMemory(data);
}

void QUASARReceiver::loadFromFiles(const Path& dataPath) {
    stats = { 0 };

    // Load camera data
    Pose cameraPose;
    Path cameraFileName = dataPath / "camera.bin";
    cameraPose.loadFromFile(cameraFileName);
    cameraPose.copyPoseToCamera(remoteCamera);
    cameraPose.copyPoseToCamera(remoteCameraWideFOV);

    // Load metadata (viewSphereDiameter and wide FOV)
    auto metadataChar = FileIO::loadFromBinaryFile(dataPath / "metadata.bin");
    std::vector<float> metadata(metadataChar.size() / sizeof(float));
    std::memcpy(metadata.data(), metadataChar.data(), metadataChar.size());

    float remoteFOVWide = metadata[0];
    float viewSphereDiameter = metadata[1];
    spdlog::debug("Loaded wide FOV: {}", remoteFOVWide);
    spdlog::debug("Loaded view sphere diameter: {}", viewSphereDiameter);

    remoteCameraWideFOV.setFovyDegrees(remoteFOVWide);
    updateViewSphere(viewSphereDiameter);

    // Read camera data
    Path colorFileName = dataPath / "color.jpg";
    atlasVideoTexture.loadFromFile(colorFileName, true, false);

    for (int layer = 0; layer < maxLayers; layer++) {
        // Load quads and depth offsets from files and decompress (nonblocking)
        double startTime = timeutils::getTimeMicros();
        frames[layer].loadFromFiles(dataPath, layer);
        stats.timeToLoadMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        startTime = timeutils::getTimeMicros();
        auto offsetsFuture = threadPool->submit_task([&]() {
            return frames[layer].decompressDepthOffsets(uncompressedOffsets);
        });
        auto quadsFuture = threadPool->submit_task([&]() {
            return frames[layer].decompressQuads(uncompressedQuads);
        });
        quadsFuture.get(); offsetsFuture.get();
        stats.timeToDecompressMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        // Copy data to GPU
        auto sizes = quadSet.loadFromMemory(uncompressedQuads, uncompressedOffsets);
        stats.timeToTransferMs += quadSet.stats.timeToTransferMs;

        // Update mesh
        const glm::vec2& gBufferSize = glm::vec2(quadSet.getSize().x, quadSet.getSize().y);
        auto& cameraToUse = getCameraToUse(layer);
        startTime = timeutils::getTimeMicros();
        meshes[layer].appendQuads(quadSet, gBufferSize);
        meshes[layer].createMeshFromProxies(quadSet, gBufferSize, cameraToUse);
        stats.timeToCreateMeshMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        auto meshBufferSizes = meshes[layer].getBufferSizes();
        stats.totalTriangles += meshBufferSizes.numIndices / 3;
        stats.sizes += sizes;
    }
}
