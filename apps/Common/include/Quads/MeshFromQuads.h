#ifndef MESH_FROM_QUADS_H
#define MESH_FROM_QUADS_H

#include <Texture.h>
#include <Primitives/Mesh.h>
#include <Cameras/PerspectiveCamera.h>
#include <Shaders/ComputeShader.h>

#include <Quads/QuadVertex.h>
#include <Quads/QuadsBuffers.h>
#include <Quads/DepthOffsets.h>

#include <Utils/TimeUtils.h>

#ifndef __ANDROID__
#define THREADS_PER_LOCALGROUP 16
#else
#define THREADS_PER_LOCALGROUP 32
#endif

namespace quasar {

#ifndef __ANDROID__
#define MAX_NUM_PROXIES (2 * 1024 * 1024)
#else
#define MAX_NUM_PROXIES (1024 * 1024)
#endif

#define VERTICES_IN_A_QUAD 4
#define INDICES_IN_A_QUAD 6
#define NUM_SUB_QUADS 4

class MeshFromQuads {
public:
    struct BufferSizes {
        uint numVertices;
        uint numIndices;
    };

    struct Stats {
        double timeToAppendQuadsMs = 0.0;
        double timeToGatherQuadsMs = 0.0;
        double timeToCreateMeshMs = 0.0;
    } stats;

    glm::uvec2 &remoteWindowSize;
    glm::uvec2 depthBufferSize;
    uint maxProxies;

    QuadBuffers currentQuadBuffers;

    MeshFromQuads(glm::uvec2 &remoteWindowSize, uint maxNumProxies = MAX_NUM_PROXIES);
    ~MeshFromQuads() = default;

    void appendQuads(
            const glm::uvec2 &gBufferSize,
            uint numProxies,
            const QuadBuffers &newQuadBuffers,
            bool refFrame = true);

    void fillQuadIndices(const glm::uvec2 &gBufferSize);

    void createMeshFromProxies(
            const glm::uvec2 &gBufferSize,
            uint numProxies,
            const DepthOffsets &depthOffsets,
            const PerspectiveCamera &remoteCamera,
            const Mesh &mesh);

    BufferSizes getBufferSizes();

private:
    Buffer meshSizesBuffer;
    Buffer prevNumProxiesBuffer;
    Buffer currNumProxiesBuffer;

    Buffer quadCreatedFlagsBuffer;
    Buffer quadIndicesBuffer;

    ComputeShader appendQuadsShader;
    ComputeShader fillQuadIndicesShader;
    ComputeShader createMeshFromQuadsShader;
};

} // namespace quasar

#endif // MESH_FROM_QUADS_H
