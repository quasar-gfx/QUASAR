#ifndef MESH_FROM_QUADS_H
#define MESH_FROM_QUADS_H

#include <Texture.h>
#include <Primitives/Mesh.h>
#include <Cameras/PerspectiveCamera.h>
#include <Shaders/ComputeShader.h>

#include <Quads/QuadsBuffers.h>
#include <Quads/DepthOffsets.h>

#include <Utils/TimeUtils.h>

#define THREADS_PER_LOCALGROUP 16

#define VERTICES_IN_A_QUAD 4
#define NUM_SUB_QUADS 4

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

    MeshFromQuads(const glm::uvec2 &remoteWindowSize);
    ~MeshFromQuads() = default;

    void createMeshFromProxies(
            unsigned int numProxies, const glm::uvec2 &depthBufferSize,
            const PerspectiveCamera &remoteCamera,
            const QuadBuffers &quadBuffers,
            const DepthOffsets &depthOffsets,
            const Texture &colorTexture,
            const Mesh &mesh);

    void appendGeometry(
            unsigned int numProxies, const glm::uvec2 &depthBufferSize,
            const PerspectiveCamera &remoteCamera,
            const QuadBuffers &quadBuffers,
            const DepthOffsets &depthOffsets,
            const Texture &colorTexture,
            const Mesh &mesh);

    void createMeshFromProxies(
            unsigned int numProxies, const glm::uvec2 &depthBufferSize,
            const PerspectiveCamera &remoteCamera,
            const QuadBuffers &quadBuffers,
            const Texture &colorTexture,
            const Mesh &mesh,
            bool appendGeometry = false);

    BufferSizes getBufferSizes();

private:
    Buffer<BufferSizes> sizesBuffer;
    Buffer<glm::ivec2> atlasOffsetBuffer;

    ComputeShader createMeshFromQuadsShader;
};

#endif // MESH_FROM_QUADS_H
