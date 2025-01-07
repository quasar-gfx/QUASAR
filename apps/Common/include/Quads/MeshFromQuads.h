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
#define INDICES_IN_A_QUAD 6
#define NUM_SUB_QUADS 4

#define MAX_NUM_PROXIES 1.5e6

class MeshFromQuads {
public:
    struct BufferSizes {
        unsigned int numVertices;
        unsigned int numIndices;
    };

    struct Stats {
        double timeToAppendProxiesMs = -1.0f;
        double timeToFillOutputQuadsMs = -1.0f;
        double timeToCreateMeshMs = -1.0f;
    } stats;

    glm::uvec2 remoteWindowSize;
    glm::uvec2 depthBufferSize;
    unsigned int maxProxies;

    unsigned int currNumProxies = 0;

    QuadBuffers currentQuadBuffers;

    MeshFromQuads(const glm::uvec2 &remoteWindowSize);
    ~MeshFromQuads() = default;

    void appendProxies(
            unsigned int numProxies,
            const QuadBuffers &newQuadBuffers,
            bool iFrame = true);

    void fillQuadIndices();

    void createMeshFromProxies(
            const glm::vec2 &gBufferSize,
            unsigned int numProxies,
            const DepthOffsets &depthOffsets,
            const PerspectiveCamera &remoteCamera,
            const Mesh &mesh,
            bool appendGeometry = false);

    BufferSizes getBufferSizes();

private:
    Buffer<BufferSizes> meshSizesBuffer;
    Buffer<unsigned int> currNumProxiesBuffer;

    Buffer<int> quadCreatedFlagsBuffer;
    Texture quadIndicesBuffer;

    ComputeShader appendProxiesShader;
    ComputeShader fillQuadIndicesShader;
    ComputeShader createMeshFromQuadsShader;
};

#endif // MESH_FROM_QUADS_H
