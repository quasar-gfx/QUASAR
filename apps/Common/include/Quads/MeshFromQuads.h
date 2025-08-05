#ifndef MESH_FROM_QUADS_H
#define MESH_FROM_QUADS_H

#include <Texture.h>
#include <Primitives/Mesh.h>
#include <Cameras/PerspectiveCamera.h>
#include <Shaders/ComputeShader.h>

#include <Quads/QuadFrame.h>
#include <Quads/QuadVertex.h>

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

    uint maxProxies;

    MeshFromQuads(const QuadFrame& quadFrame, uint maxNumProxies = MAX_NUM_PROXIES);
    ~MeshFromQuads() = default;

    void appendQuads(const QuadFrame& quadFrame, const glm::vec2& gBufferSize, bool isRefFrame = true);
    void createMeshFromProxies(const QuadFrame& quadFrame, const glm::vec2& gBufferSize, const PerspectiveCamera& remoteCamera, const Mesh& mesh);

    BufferSizes getBufferSizes() const;

private:
    QuadBuffers currentQuadBuffers;

    Buffer meshSizesBuffer;
    Buffer prevNumProxiesBuffer;
    Buffer currNumProxiesBuffer;

    Buffer quadCreatedFlagsBuffer;
    Buffer quadIndicesMap;

    ComputeShader appendQuadsShader;
    ComputeShader fillQuadIndicesShader;
    ComputeShader createMeshFromQuadsShader;

    void fillQuadIndices(const QuadFrame& quadFrame, const glm::vec2& gBufferSize);
};

} // namespace quasar

#endif // MESH_FROM_QUADS_H
