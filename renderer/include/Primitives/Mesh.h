#ifndef MESH_H
#define MESH_H

#include <vector>

#include <Vertex.h>
#include <Buffer.h>
#include <Shaders/Shader.h>
#include <Texture.h>
#include <Primitives/Entity.h>
#include <Materials/Material.h>
#include <Scene.h>
#include <Cameras/PerspectiveCamera.h>
#include <Cameras/VRCamera.h>

namespace quasar {

struct MeshDataCreateParams {
    const void* verticesData;
    size_t verticesSize;
    const uint* indicesData = nullptr;
    size_t indicesSize = 0;
    uint vertexSize = sizeof(Vertex);
    VertexInputAttributes attributes = Vertex::getVertexInputAttributes();
    Material* material;
    float IBL = 1.0;
    GLenum usage = GL_STATIC_DRAW;
    bool indirectDraw = false;
};

struct MeshSizeCreateParams {
    uint maxVertices;
    uint maxIndices = 0;
    uint vertexSize = sizeof(Vertex);
    VertexInputAttributes attributes = Vertex::getVertexInputAttributes();
    Material* material;
    float IBL = 1.0;
    GLenum usage = GL_STATIC_DRAW;
    bool indirectDraw = false;
};

struct DrawElementsIndirectCommand {
    GLuint count = 0;
    GLuint instanceCount = 1;
    GLuint firstIndex = 0;
    GLuint baseVertex = 0;
    GLuint baseInstance = 0;
};

class Mesh : public Entity {
public:
    Buffer vertexBuffer;
    Buffer indexBuffer;
    Buffer indirectBuffer;

    uint vertexSize;
    VertexInputAttributes attributes;

    Material* material;

    float IBL = 1.0;

    GLenum usage;

    bool indirectDraw = false;

    Mesh();
    Mesh(const MeshDataCreateParams& params);
    Mesh(const MeshSizeCreateParams& params);

    virtual void bindMaterial(Scene& scene, const glm::mat4& model,
                              const Material* overrideMaterial = nullptr, const Texture* prevIDMap = nullptr) override;

    virtual RenderStats draw(GLenum primativeType, const Camera& camera, const glm::mat4& model,
                             bool frustumCull = true, const Material* overrideMaterial = nullptr) override;
    virtual RenderStats draw(GLenum primativeType, const Camera& camera, const glm::mat4& model,
                             const BoundingSphere& boundingSphere, const Material* overrideMaterial = nullptr) override;
    virtual RenderStats draw(GLenum primativeType);

    void setBuffers(const void* vertices, uint verticesSize, const uint* indices = nullptr, uint indicesSize = 0);
    void setBuffers(uint verticesSize, uint indicesSize);

    void resizeBuffers(uint verticesSize, uint indicesSize);
    void updateAABB(const void* vertices, uint verticesSize);

    EntityType getType() const override { return EntityType::MESH; }

protected:
    GLuint vertexArrayBuffer;

    void setArrayBufferAttributes(const VertexInputAttributes& attributes, uint vertexSize);

    void setMaterialCameraParams(const Camera& camera, const Material* material);
};

} // namespace quasar

#endif // MESH_H
