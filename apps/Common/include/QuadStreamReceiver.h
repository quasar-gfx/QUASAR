#ifndef QUASAR_RECEIVER_H
#define QUASAR_RECEIVER_H

#include <Path.h>
#include <Quads/QuadSet.h>
#include <Quads/QuadFrames.h>
#include <Quads/QuadMesh.h>
#include <Quads/QuadMaterial.h>

namespace quasar {

class QuadStreamReceiver {
public:
    struct Stats {
        uint totalTriangles = 0;
        double loadFromFilesTime = 0.0;
        double decompressTime = 0.0;
        double transferTime = 0.0;
        double createMeshTime = 0.0;
        QuadSet::Sizes sizes{};
    } stats;

    uint maxViews;
    std::vector<ReferenceFrame> frames;

    QuadStreamReceiver(QuadSet& quadSet, uint maxViews, float remoteFOV, float remoteFOVWide, float viewBoxSize)
        : quadSet(quadSet)
        , maxViews(maxViews)
        , frames(maxViews)
        , uncompressedQuads(sizeof(uint) + quadSet.quadBuffers.maxProxies * sizeof(QuadMapDataPacked))
        , uncompressedOffsets(quadSet.depthOffsets.getSize().x * quadSet.depthOffsets.getSize().y * 4 * sizeof(uint16_t))
    {
        remoteCameras.reserve(maxViews);
        for (int view = 0; view < maxViews; view++) {
            remoteCameras.emplace_back(quadSet.getSize());
            remoteCameras[view].setFovyDegrees(remoteFOV);
        }

        PerspectiveCamera& remoteCameraCenter = remoteCameras[0];
        for (int view = 1; view < maxViews - 1; view++) {
            const glm::vec3& offset = offsets[view - 1];
            const glm::vec3& right = remoteCameraCenter.getRightVector();
            const glm::vec3& up = remoteCameraCenter.getUpVector();
            const glm::vec3& forward = remoteCameraCenter.getForwardVector();

            glm::vec3 worldOffset =
                right   * offset.x * viewBoxSize / 2.0f +
                up      * offset.y * viewBoxSize / 2.0f +
                forward * -offset.z * viewBoxSize / 2.0f;

            remoteCameras[view].setViewMatrix(remoteCameraCenter.getViewMatrix());
            remoteCameras[view].setPosition(remoteCameraCenter.getPosition() + worldOffset);
            remoteCameras[view].updateViewMatrix();
        }

        // Last camera has wider FOV
        remoteCameras[maxViews-1].setFovyDegrees(remoteFOVWide);
        remoteCameras[maxViews-1].setViewMatrix(remoteCameraCenter.getViewMatrix());

        TextureDataCreateParams texParams = {
            .internalFormat = GL_RGB,
            .format = GL_RGB,
            .wrapS = GL_REPEAT,
            .wrapT = GL_REPEAT,
            .minFilter = GL_NEAREST,
            .magFilter = GL_NEAREST,
        };
        colorTextures.reserve(maxViews);
        meshes.reserve(maxViews);
        for (int view = 0; view < maxViews; view++) {
            colorTextures.emplace_back(texParams);
            // We can use less vertices and indicies for the additional views since they will be sparser
            uint maxProxies = (view == 0 || view == maxViews - 1) ? MAX_QUADS_PER_MESH : MAX_QUADS_PER_MESH / 4;
            meshes.emplace_back(quadSet, colorTextures[view], maxProxies);
        }
    }
    ~QuadStreamReceiver() = default;

    QuadMesh& getMesh(int view) {
        return meshes[view];
    }

    PerspectiveCamera& getRemoteCamera(int view = 0) {
        return remoteCameras[view];
    }

    void loadFromFiles(const Path& dataPath) {
        stats = {};
        for (int view = 0; view < maxViews; view++) {
            // Load texture
            Path colorFileName = dataPath / ("color" + std::to_string(view) + ".jpg");
            colorTextures[view].loadFromFile(colorFileName, true, false);

            // Load quads and depth offsets from files and decompress (nonblocking)
            double startTime = timeutils::getTimeMicros();
            frames[view].loadFromFiles(dataPath, view);
            stats.loadFromFilesTime += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

            startTime = timeutils::getTimeMicros();
            auto offsetsFuture = frames[view].decompressDepthOffsets(uncompressedOffsets);
            auto quadsFuture = frames[view].decompressQuads(uncompressedQuads);
            quadsFuture.get(); offsetsFuture.get();
            stats.decompressTime += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

            // Copy data to GPU
            auto sizes = quadSet.copyFromCPU(uncompressedQuads, uncompressedOffsets);
            stats.transferTime += quadSet.stats.timeToTransferMs;

            // Update mesh
            const glm::vec2& gBufferSize = glm::vec2(colorTextures[view].width, colorTextures[view].height);
            startTime = timeutils::getTimeMicros();
            meshes[view].appendQuads(quadSet, gBufferSize);
            meshes[view].createMeshFromProxies(quadSet, gBufferSize, remoteCameras[view]);
            stats.createMeshTime += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

            auto meshBufferSizes = meshes[view].getBufferSizes();
            stats.totalTriangles += meshBufferSizes.numIndices / 3;
            stats.sizes += sizes;
        }
    }

private:
    const std::vector<glm::vec3> offsets = {
        glm::vec3(-1.0f, +1.0f, -1.0f), // Top-left
        glm::vec3(+1.0f, +1.0f, -1.0f), // Top-right
        glm::vec3(+1.0f, -1.0f, -1.0f), // Bottom-right
        glm::vec3(-1.0f, -1.0f, -1.0f), // Bottom-left
        glm::vec3(-1.0f, +1.0f, +1.0f), // Top-left
        glm::vec3(+1.0f, +1.0f, +1.0f), // Top-right
        glm::vec3(+1.0f, -1.0f, +1.0f), // Bottom-right
        glm::vec3(-1.0f, -1.0f, +1.0f), // Bottom-left
    };

    QuadSet& quadSet;

    std::vector<PerspectiveCamera> remoteCameras;

    QuadMaterial quadMaterial;
    std::vector<Texture> colorTextures;
    std::vector<QuadMesh> meshes;

    // Temporary buffers for decompression
    std::vector<char> uncompressedQuads, uncompressedOffsets;
};

} // namespace quasar

#endif // QUASAR_RECEIVER_H
