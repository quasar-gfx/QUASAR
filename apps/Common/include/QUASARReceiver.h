#ifndef QUASAR_RECEIVER_H
#define QUASAR_RECEIVER_H

#include <Path.h>
#include <Quads/QuadSet.h>
#include <Quads/QuadFrames.h>
#include <Quads/QuadMesh.h>
#include <Quads/QuadMaterial.h>

namespace quasar {

class QUASARReceiver {
public:
    struct Stats {
        uint totalTriangles = 0;
        double loadFromFilesTime = 0.0;
        double decompressTime = 0.0;
        double transferTime = 0.0;
        double createMeshTime = 0.0;
        QuadSet::Sizes sizes{};
    } stats;

    uint maxLayers;
    std::vector<ReferenceFrame> frames;

    QUASARReceiver(QuadSet& quadSet, uint maxLayers, float remoteFOV, float remoteFOVWide)
        : quadSet(quadSet)
        , maxLayers(maxLayers)
        , remoteCamera(quadSet.getSize())
        , remoteCameraWideFov(quadSet.getSize())
        , frames(maxLayers)
        , uncompressedQuads(sizeof(uint) + quadSet.quadBuffers.maxProxies * sizeof(QuadMapDataPacked))
        , uncompressedOffsets(quadSet.depthOffsets.getSize().x * quadSet.depthOffsets.getSize().y * 4 * sizeof(uint16_t))
    {
        remoteCamera.setFovyDegrees(remoteFOV);

        // Camera has wider FOV
        remoteCameraWideFov.setFovyDegrees(remoteFOVWide);
        remoteCameraWideFov.setViewMatrix(remoteCamera.getViewMatrix());

        TextureDataCreateParams texParams = {
            .internalFormat = GL_RGB,
            .format = GL_RGB,
            .wrapS = GL_REPEAT,
            .wrapT = GL_REPEAT,
            .minFilter = GL_NEAREST,
            .magFilter = GL_NEAREST,
        };
        colorTextures.reserve(maxLayers);
        meshes.reserve(maxLayers);
        for (int layer = 0; layer < maxLayers; layer++) {
            colorTextures.emplace_back(texParams);
            // First and last layer need a lot of quads, each subsequent one has less
            uint maxProxies = (layer == 0 || layer == maxLayers - 1) ? MAX_QUADS_PER_MESH : MAX_QUADS_PER_MESH / 4;
            meshes.emplace_back(quadSet, colorTextures[layer], maxProxies);
        }
    }
    ~QUASARReceiver() = default;

    QuadMesh& getMesh(int layer) {
        return meshes[layer];
    }

    PerspectiveCamera& getRemoteCamera() {
        return remoteCamera;
    }
    PerspectiveCamera& getRemoteCameraWideFov() {
        return remoteCameraWideFov;
    }

    void loadFromFiles(const Path& dataPath) {
        stats = {};
        for (int layer = 0; layer < maxLayers; layer++) {
            // Load texture
            Path colorFileName = dataPath / ("color" + std::to_string(layer) + ".jpg");
            colorTextures[layer].loadFromFile(colorFileName, true, false);

            // Load quads and depth offsets from files and decompress (nonblocking)
            double startTime = timeutils::getTimeMicros();
            frames[layer].loadFromFiles(dataPath, layer);
            stats.loadFromFilesTime += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

            startTime = timeutils::getTimeMicros();
            auto offsetsFuture = frames[layer].decompressDepthOffsets(uncompressedOffsets);
            auto quadsFuture = frames[layer].decompressQuads(uncompressedQuads);
            quadsFuture.get(); offsetsFuture.get();
            stats.decompressTime += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

            // Copy data to GPU
            auto sizes = quadSet.copyFromCPU(uncompressedQuads, uncompressedOffsets);
            stats.transferTime += quadSet.stats.timeToTransferMs;

            // Update mesh
            const glm::vec2& gBufferSize = glm::vec2(colorTextures[layer].width, colorTextures[layer].height);
            auto& cameraToUse = getCameraToUse(layer);
            startTime = timeutils::getTimeMicros();
            meshes[layer].appendQuads(quadSet, gBufferSize);
            meshes[layer].createMeshFromProxies(quadSet, gBufferSize, cameraToUse);
            stats.createMeshTime += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

            auto meshBufferSizes = meshes[layer].getBufferSizes();
            stats.totalTriangles += meshBufferSizes.numIndices / 3;
            stats.sizes += sizes;
        }
    }

private:
    QuadSet& quadSet;

    PerspectiveCamera remoteCamera;
    PerspectiveCamera remoteCameraWideFov;

    QuadMaterial quadMaterial;
    std::vector<Texture> colorTextures;
    std::vector<QuadMesh> meshes;

    // Temporary buffers for decompression
    std::vector<char> uncompressedQuads, uncompressedOffsets;

    inline const PerspectiveCamera& getCameraToUse(int layer) const {
        return (layer == maxLayers - 1) ? remoteCameraWideFov : remoteCamera;
    }
};

} // namespace quasar

#endif // QUASAR_RECEIVER_H
