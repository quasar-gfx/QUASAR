#ifndef MESH_FROM_QUADS_H
#define MESH_FROM_QUADS_H

#include <Texture.h>
#include <Primitives/Mesh.h>
#include <Cameras/PerspectiveCamera.h>
#include <Shaders/ComputeShader.h>

#include <QuadsGenerator.h>
#include <QuadMaterial.h>

#include <Utils/TimeUtils.h>

#define THREADS_PER_LOCALGROUP 16

#define VERTICES_IN_A_QUAD 4
#define NUM_SUB_QUADS 4

#define DEFAULT_ATLAS_SIZE glm::uvec2(4096, 4096)

class MeshFromQuads {
public:
    struct BufferSizes {
        unsigned int numVertices;
        unsigned int numIndices;
    };

    struct Stats {
        double timeToCreateMeshMs = -1.0f;
    } stats;

    glm::uvec2 remoteWindowSize;
    glm::uvec2 atlasSize;

    Texture atlas;

    MeshFromQuads(const glm::uvec2 &remoteWindowSize, const glm::uvec2 &atlasSize = DEFAULT_ATLAS_SIZE);

    void createMeshFromProxies(
            unsigned int numProxies, const glm::uvec2 &depthBufferSize,
            const PerspectiveCamera &remoteCamera,
            const Buffer<unsigned int> &outputNormalSphericalsBuffer,
            const Buffer<float> &outputDepthsBuffer,
            const Buffer<unsigned int> &outputXYsBuffer,
            const Buffer<unsigned int> &outputOffsetSizeFlattenedsBuffer,
            const Texture& depthOffsetsBuffer,
            const Texture& colorTexture,
            const Mesh& mesh);

    void appendGeometry(
            unsigned int numProxies, const glm::uvec2 &depthBufferSize,
            const PerspectiveCamera &remoteCamera,
            const Buffer<unsigned int> &outputNormalSphericalsBuffer,
            const Buffer<float> &outputDepthsBuffer,
            const Buffer<unsigned int> &outputXYsBuffer,
            const Buffer<unsigned int> &outputOffsetSizeFlattenedsBuffer,
            const Texture& depthOffsetsBuffer,
            const Texture& colorTexture,
            const Mesh& mesh);

    void createMeshFromProxies(
            unsigned int numProxies, const glm::uvec2 &depthBufferSize,
            const PerspectiveCamera &remoteCamera,
            const Buffer<unsigned int> &outputNormalSphericalsBuffer,
            const Buffer<float> &outputDepthsBuffer,
            const Buffer<unsigned int> &outputXYsBuffer,
            const Buffer<unsigned int> &outputOffsetSizeFlattenedsBuffer,
            const Texture& colorTexture,
            const Mesh& mesh,
            bool appendGeometry = false);

    BufferSizes getBufferSizes() {
        sizesBuffer.bind();
        sizesBuffer.getData(&bufferSizes);
        return bufferSizes;
    }

private:
    BufferSizes bufferSizes = { 0 };
    Buffer<BufferSizes> sizesBuffer;

    ComputeShader createMeshFromQuadsShader;
};

#endif // MESH_FROM_QUADS_H
