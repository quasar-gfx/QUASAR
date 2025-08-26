#ifndef QUADS_RECEIVER_H
#define QUADS_RECEIVER_H

#include <Path.h>
#include <Quads/QuadSet.h>
#include <Quads/QuadFrames.h>
#include <Quads/QuadMesh.h>
#include <Quads/QuadMaterial.h>

namespace quasar {

class QuadsReceiver {
public:
    struct Stats {
        uint totalTriangles = 0;
        double loadFromFilesTime = 0.0;
        double decompressTime = 0.0;
        double transferTime = 0.0;
        double createMeshTime = 0.0;
        QuadSet::Sizes sizes{};
    } stats;

    ReferenceFrame frame;

    QuadsReceiver(QuadSet& quadSet, float remoteFOV)
        : quadSet(quadSet)
        , remoteCamera(quadSet.getSize())
        , colorTexture({
            .internalFormat = GL_RGB,
            .format = GL_RGB,
            .wrapS = GL_REPEAT,
            .wrapT = GL_REPEAT,
            .minFilter = GL_NEAREST,
            .magFilter = GL_NEAREST,
        })
        , mesh(quadSet, colorTexture)
        , uncompressedQuads(sizeof(uint) + quadSet.quadBuffers.maxProxies * sizeof(QuadMapDataPacked))
        , uncompressedOffsets(quadSet.depthOffsets.getSize().x * quadSet.depthOffsets.getSize().y * 4 * sizeof(uint16_t))
    {
        remoteCamera.setFovyDegrees(remoteFOV);
    }
    ~QuadsReceiver() = default;

    QuadMesh& getMesh() {
        return mesh;
    }

    PerspectiveCamera& getRemoteCamera() {
        return remoteCamera;
    }

    void loadFromFiles(const Path& dataPath) {
        stats = {};

        // Load texture
        std::string colorFileName = dataPath / "color.jpg";
        colorTexture.loadFromFile(colorFileName, true, false);

        // Load quads and depth offsets from files and decompress (nonblocking)
        double startTime = timeutils::getTimeMicros();
        frame.loadFromFiles(dataPath);
        stats.loadFromFilesTime += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        startTime = timeutils::getTimeMicros();
        auto offsetsFuture = frame.decompressDepthOffsets(uncompressedOffsets);
        auto quadsFuture = frame.decompressQuads(uncompressedQuads);
        quadsFuture.get(); offsetsFuture.get();
        stats.decompressTime += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        // Copy data to GPU
        auto sizes = quadSet.copyFromCPU(uncompressedQuads, uncompressedOffsets);
        stats.transferTime += quadSet.stats.timeToTransferMs;

        // Update mesh
        const glm::vec2& gBufferSize = glm::vec2(colorTexture.width, colorTexture.height);
        startTime = timeutils::getTimeMicros();
        mesh.appendQuads(quadSet, gBufferSize);
        mesh.createMeshFromProxies(quadSet, gBufferSize, remoteCamera);
        stats.createMeshTime += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        auto meshBufferSizes = mesh.getBufferSizes();
        stats.totalTriangles += meshBufferSizes.numIndices / 3;
        stats.sizes += sizes;
    }

private:
    QuadSet& quadSet;

    PerspectiveCamera remoteCamera;

    QuadMaterial quadMaterial;
    Texture colorTexture;
    QuadMesh mesh;

    // Temporary buffers for decompression
    std::vector<char> uncompressedQuads, uncompressedOffsets;
};

} // namespace quasar

#endif // QUADS_RECEIVER_H
