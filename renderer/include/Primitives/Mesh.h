#ifndef MESH_H
#define MESH_H

#include <vector>

#include <Vertex.h>
#include <Buffer.h>
#include <Shaders/Shader.h>
#include <Texture.h>
#include <Primitives/Entity.h>
#include <Scene.h>
#include <Cameras/PerspectiveCamera.h>
#include <Cameras/VRCamera.h>

namespace quasar {

struct MeshDataCreateParams {
    const void* verticesData;
    size_t verticesSize;
    const uint* indicesData = nullptr;
    size_t indicesSize = 0;
    size_t vertexSize = sizeof(Vertex);
    VertexInputAttributes attributes = Vertex::getVertexInputAttributes();
    const Material* material;
    float IBL = 1.0;
    GLenum usage = GL_STATIC_DRAW;
    bool indirectDraw = false;
};

struct MeshSizeCreateParams {
    size_t maxVertices;
    size_t maxIndices = 0;
    size_t vertexSize = sizeof(Vertex);
    VertexInputAttributes attributes = Vertex::getVertexInputAttributes();
    const Material* material;
    float IBL = 1.0;
    GLenum usage = GL_STATIC_DRAW;
    bool indirectDraw = false;
};

struct DrawElementsIndirectCommand {
    uint32_t count = 0;
    uint32_t instanceCount = 1;
    uint32_t firstIndex = 0;
    uint32_t baseVertex = 0;
    uint32_t baseInstance = 0;
};

class Mesh : public Entity {
public:
    Buffer vertexBuffer;
    Buffer indexBuffer;
    Buffer indirectBuffer;
    bool indirectDraw;

    float IBL;

    Mesh();
    Mesh(const MeshDataCreateParams& params);
    Mesh(const MeshSizeCreateParams& params);

    virtual void bindMaterial(Scene& scene, Buffer& pointLightsUBO,
                              const Material* overrideMaterial = nullptr, const Texture* prevIDMap = nullptr) override;

    virtual RenderStats draw(GLenum primitiveType, const Camera& camera, const glm::mat4& model,
                             bool frustumCull = true, const Material* overrideMaterial = nullptr) override;
    virtual RenderStats draw(GLenum primitiveType, const Camera& camera, const glm::mat4& model,
                             const BoundingSphere& boundingSphere, const Material* overrideMaterial = nullptr) override;
    virtual RenderStats draw(GLenum primitiveType);

    void setBuffers(const void* vertices, uint verticesSize, const uint* indices = nullptr, uint indicesSize = 0);
    void setBuffers(uint verticesSize, uint indicesSize);

    void resizeBuffers(uint verticesSize, uint indicesSize);

    void updateAABB(const glm::vec3& min, const glm::vec3& max);
    void updateAABB(const void* vertices, uint verticesSize);

protected:
    GLenum usage;
    uint vertexSize;

    GLuint vertexArrayBuffer;
    VertexInputAttributes attributes;

    void setArrayBufferAttributes(const VertexInputAttributes& attributes, uint vertexSize);

    void setMaterialCameraParams(const Camera& camera, const Material* material);
};

} // namespace quasar

#endif // MESH_H
