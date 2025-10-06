#ifndef QUAD_MESH_H
#define QUAD_MESH_H

#include <Texture.h>
#include <Primitives/Mesh.h>
#include <Cameras/PerspectiveCamera.h>
#include <Shaders/ComputeShader.h>

#include <Quads/QuadSet.h>
#include <Quads/QuadVertex.h>
#include <Quads/QuadMaterial.h>

namespace quasar {

#define MAX_PROXIES_PER_MESH 640000u

#define VERTICES_IN_A_QUAD 4
#define INDICES_IN_A_QUAD  6
#define NUM_SUB_QUADS      4

class QuadMesh : public Mesh {
public:
    struct BufferSizes {
        uint32_t numVertices;
        uint32_t numIndices;
        uint32_t numIndicesTransparent;
    };

    enum class DrawState {
        OPAQUE,
        TRANSPARENT,
        BOTH
    };

    struct Stats {
        double appendQuadsTimeMs = 0.0;
        double createMeshTimeMs = 0.0;
    } stats;

    QuadMesh(const QuadSet& quadSet, Texture& colorTexture, Texture& alphaTexture, uint maxProxies = MAX_PROXIES_PER_MESH);
    QuadMesh(const QuadSet& quadSet, Texture& colorTexture, Texture& alphaTexture, const glm::vec4& textureExtent, uint maxProxies = MAX_PROXIES_PER_MESH);
    ~QuadMesh() = default;

    glm::vec4 getTextureExtent() const { return textureExtent; }
    uint32_t getMaxProxies() const { return maxProxies; }

    void setTextureExtent(const glm::vec4& extent) { textureExtent = extent; }
    void setExpandQuadAmount(float amount) { expandQuadAmount = amount; }
    void setDrawState(DrawState drawState) { this->drawState = drawState; }

    void appendQuads(const QuadSet& quadSet, const glm::vec2& gBufferSize, bool isFullFrame = true);
    void createMeshFromProxies(const QuadSet& quadSet, const glm::vec2& gBufferSize, const PerspectiveCamera& remoteCamera);

    BufferSizes getBufferSizes() const;

    RenderStats draw(GLenum primitiveType) override;

private:
    uint32_t maxProxies;

    DrawState drawState = DrawState::BOTH;
    float expandQuadAmount = 0.025f;

    glm::vec4 textureExtent;

    uint32_t currNumProxies;
    uint32_t currNumProxiesTransparent;

    QuadBuffers currentQuadBuffers;

    Buffer sizesBuffer;

    Buffer quadIndexMap;
    Buffer quadCreatedFlags;

    Buffer indexBufferTransparent;
    Buffer indirectBufferTransparent;

    ComputeShader appendQuadsShader;
    ComputeShader createQuadMeshShader;
};

} // namespace quasar

#endif // QUAD_MESH_H
