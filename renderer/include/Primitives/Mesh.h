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

    Mesh() : vertexBuffer(GL_ARRAY_BUFFER, sizeof(Vertex))
            , indexBuffer(GL_ELEMENT_ARRAY_BUFFER, sizeof(uint)) {
        setArrayBufferAttributes(Vertex::getVertexInputAttributes(), sizeof(Vertex));
    }
    Mesh(const MeshDataCreateParams& params)
            : material(params.material)
            , IBL(params.IBL)
            , usage(params.usage)
            , vertexBuffer(GL_ARRAY_BUFFER, sizeof(Vertex), params.usage)
            , indexBuffer(GL_ELEMENT_ARRAY_BUFFER, sizeof(uint), params.usage)
            , indirectDraw(params.indirectDraw)
            , indirectBuffer(GL_DRAW_INDIRECT_BUFFER, sizeof(DrawElementsIndirectCommand), params.usage)
            , vertexSize(params.vertexSize)
            , attributes(params.attributes) {
        setArrayBufferAttributes(params.attributes, params.vertexSize);
        setBuffers(params.verticesData, params.verticesSize, params.indicesData, params.indicesSize);

        if (indirectDraw) {
            indirectBuffer.bind();
            DrawElementsIndirectCommand indirectCommand;
            indirectBuffer.setData(sizeof(DrawElementsIndirectCommand), &indirectCommand);
            indirectBuffer.unbind();
        }
    }
    Mesh(const MeshSizeCreateParams& params)
            : material(params.material)
            , IBL(params.IBL)
            , usage(params.usage)
            , vertexBuffer(GL_ARRAY_BUFFER, sizeof(Vertex), params.usage)
            , indexBuffer(GL_ELEMENT_ARRAY_BUFFER, sizeof(uint), params.usage)
            , indirectDraw(params.indirectDraw)
            , indirectBuffer(GL_DRAW_INDIRECT_BUFFER, sizeof(DrawElementsIndirectCommand), params.usage)
            , vertexSize(params.vertexSize)
            , attributes(params.attributes) {
        setArrayBufferAttributes(params.attributes, params.vertexSize);
        setBuffers(params.maxVertices, params.maxIndices);

        if (indirectDraw) {
            indirectBuffer.bind();
            DrawElementsIndirectCommand indirectCommand;
            indirectBuffer.setData(sizeof(DrawElementsIndirectCommand), &indirectCommand);
            indirectBuffer.unbind();
        }
    }

    virtual void bindMaterial(const Scene& scene, const glm::mat4& model,
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
