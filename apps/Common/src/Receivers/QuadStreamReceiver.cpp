#include <Receivers/QuadStreamReceiver.h>

using namespace quasar;

QuadStreamReceiver::QuadStreamReceiver(QuadSet& quadSet, uint maxViews)
    : quadSet(quadSet)
    , maxViews(maxViews)
    , frames(maxViews)
    , uncompressedQuads(sizeof(uint) + quadSet.quadBuffers.maxProxies * sizeof(QuadMapDataPacked))
    , uncompressedOffsets(quadSet.depthOffsets.getSize().x * quadSet.depthOffsets.getSize().y * 4 * sizeof(uint16_t))
{
    TextureDataCreateParams texParams = {
        .internalFormat = GL_RGB,
        .format = GL_RGB,
        .wrapS = GL_REPEAT,
        .wrapT = GL_REPEAT,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    };
    remoteCameras.reserve(maxViews);
    colorTextures.reserve(maxViews);
    meshes.reserve(maxViews);
    for (int view = 0; view < maxViews; view++) {
        remoteCameras.emplace_back(quadSet.getSize());

        colorTextures.emplace_back(texParams);

        // We can use less vertices and indicies for the additional views since they will be sparser
        uint maxProxies = (view == 0 || view == maxViews - 1) ? MAX_QUADS_PER_MESH : MAX_QUADS_PER_MESH / 8;
        meshes.emplace_back(quadSet, colorTextures[view], maxProxies);
    }

    threadPool = std::make_unique<BS::thread_pool<>>(2);
}

QuadStreamReceiver::QuadStreamReceiver(QuadSet& quadSet, uint maxViews, float remoteFOV, float remoteFOVWide, float viewBoxSize)
    : QuadStreamReceiver(quadSet, maxViews)
{
    for (int view = 0; view < maxViews; view++) {
        remoteCameras[view].setFovyDegrees(remoteFOV);
    }
    // Last camera has different FOV
    remoteCameras[maxViews-1].setFovyDegrees(remoteFOVWide);

    updateViewBox(viewBoxSize);
}

void QuadStreamReceiver::updateViewBox(float viewBoxSize) {
    this->viewBoxSize = viewBoxSize;

    PerspectiveCamera& remoteCameraCenter = remoteCameras[0];

    // Update other cameras in view box corners
    for (int view = 1; view < maxViews - 1; view++) {
        const glm::vec3& offset = offsets[view - 1];
        const glm::vec3& right = remoteCameraCenter.getRightVector();
        const glm::vec3& up = remoteCameraCenter.getUpVector();
        const glm::vec3& forward = remoteCameraCenter.getForwardVector();

        glm::vec3 worldOffset =
            right   * +offset.x * viewBoxSize / 2.0f +
            up      * +offset.y * viewBoxSize / 2.0f +
            forward * -offset.z * viewBoxSize / 2.0f;

        remoteCameras[view].setViewMatrix(remoteCameraCenter.getViewMatrix());
        remoteCameras[view].setPosition(remoteCameraCenter.getPosition() + worldOffset);
        remoteCameras[view].updateViewMatrix();
    }

    // Update wide fov camera
    remoteCameras[maxViews-1].setViewMatrix(remoteCameraCenter.getViewMatrix());
}

void QuadStreamReceiver::loadFromFiles(const Path& dataPath) {
    stats = { 0 };

    // Load camera data
    Pose cameraPose;
    PerspectiveCamera& remoteCameraCenter = remoteCameras[0];
    Path cameraFileName = dataPath / "camera.bin";
    cameraPose.loadFromFile(cameraFileName);
    cameraPose.copyPoseToCamera(remoteCameraCenter);
    for (int view = 1; view < maxViews; view++) {
        remoteCameras[view].setViewMatrix(remoteCameraCenter.getViewMatrix());
        cameraPose.copyPoseToCamera(remoteCameras[view]);
    }

    // Load metadata (viewBoxSize and wide FOV)
    auto metadataChar = FileIO::loadFromBinaryFile(dataPath / "metadata.bin");
    std::vector<float> metadata(metadataChar.size() / sizeof(float));
    std::memcpy(metadata.data(), metadataChar.data(), metadataChar.size());

    float remoteFOVWide = metadata[0];
    float viewBoxSize = metadata[1];
    spdlog::debug("Loaded wide FOV: {}", remoteFOVWide);
    spdlog::debug("Loaded view box size: {}", viewBoxSize);

    PerspectiveCamera& remoteCameraWideFOV = remoteCameras[maxViews-1];
    remoteCameraWideFOV.setFovyDegrees(remoteFOVWide);
    updateViewBox(viewBoxSize);

    for (int view = 0; view < maxViews; view++) {
        // Load texture
        Path colorFileName = (dataPath / ("color" + std::to_string(view))).withExtension(".jpg");
        colorTextures[view].loadFromFile(colorFileName, true, false);

        // Load quads and depth offsets from files
        double startTime = timeutils::getTimeMicros();
        frames[view].loadFromFiles(dataPath, view);
        stats.timeToLoadMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        // Decompress (asynchronous)
        startTime = timeutils::getTimeMicros();
        auto offsetsFuture = threadPool->submit_task([&]() {
            return frames[view].decompressDepthOffsets(uncompressedOffsets);
        });
        auto quadsFuture = threadPool->submit_task([&]() {
            return frames[view].decompressQuads(uncompressedQuads);
        });
        quadsFuture.get(); offsetsFuture.get();
        stats.timeToDecompressMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        // Copy data to GPU
        auto sizes = quadSet.loadFromMemory(uncompressedQuads, uncompressedOffsets);
        stats.timeToTransferMs += quadSet.stats.timeToTransferMs;

        // Update mesh
        const glm::vec2& gBufferSize = glm::vec2(colorTextures[view].width, colorTextures[view].height);
        startTime = timeutils::getTimeMicros();
        meshes[view].appendQuads(quadSet, gBufferSize);
        meshes[view].createMeshFromProxies(quadSet, gBufferSize, remoteCameras[view]);
        stats.timeToCreateMeshMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        auto meshBufferSizes = meshes[view].getBufferSizes();
        stats.totalTriangles += meshBufferSizes.numIndices / 3;
        stats.sizes += sizes;
    }
}
