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

struct MeshDataCreateParams {
    const void* verticesData;
    size_t verticesSize;
    const unsigned int* indicesData = nullptr;
    size_t indicesSize = 0;
    unsigned int vertexSize = sizeof(Vertex);
    VertexInputAttributes attributes = Vertex::getVertexInputAttributes();
    Material* material;
    float IBL = 1.0;
    GLenum usage = GL_STATIC_DRAW;
    bool indirectDraw = false;
};

struct MeshSizeCreateParams {
    unsigned int maxVertices;
    unsigned int maxIndices = 0;
    unsigned int vertexSize = sizeof(Vertex);
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

    unsigned int vertexSize;
    VertexInputAttributes attributes;

    Material* material;

    float IBL = 1.0;

    GLenum usage;

    bool indirectDraw = false;

    Mesh() : vertexBuffer(GL_ARRAY_BUFFER, GL_STATIC_DRAW, sizeof(Vertex))
            , indexBuffer(GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW, sizeof(unsigned int)) {
        setArrayBufferAttributes(Vertex::getVertexInputAttributes(), sizeof(Vertex));
    }
    Mesh(const MeshDataCreateParams &params)
            : material(params.material)
            , IBL(params.IBL)
            , usage(params.usage)
            , vertexBuffer(GL_ARRAY_BUFFER, params.usage, sizeof(Vertex))
            , indexBuffer(GL_ELEMENT_ARRAY_BUFFER, params.usage, sizeof(unsigned int))
            , indirectDraw(params.indirectDraw)
            , indirectBuffer(GL_DRAW_INDIRECT_BUFFER, params.usage, sizeof(DrawElementsIndirectCommand))
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
    Mesh(const MeshSizeCreateParams &params)
            : material(params.material)
            , IBL(params.IBL)
            , usage(params.usage)
            , vertexBuffer(GL_ARRAY_BUFFER, params.usage, sizeof(Vertex))
            , indexBuffer(GL_ELEMENT_ARRAY_BUFFER, params.usage, sizeof(unsigned int))
            , indirectDraw(params.indirectDraw)
            , indirectBuffer(GL_DRAW_INDIRECT_BUFFER, params.usage, sizeof(DrawElementsIndirectCommand))
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

    virtual void bindMaterial(const Scene &scene, const glm::mat4 &model,
                              const Material* overrideMaterial = nullptr, const Texture* prevDepthMap = nullptr) override;

    virtual RenderStats draw(GLenum primativeType, const Camera &camera, const glm::mat4 &model,
                             bool frustumCull = true, const Material* overrideMaterial = nullptr) override;
    virtual RenderStats draw(GLenum primativeType, const Camera &camera, const glm::mat4 &model,
                             const BoundingSphere &boundingSphere, const Material* overrideMaterial = nullptr) override;
    virtual RenderStats draw(GLenum primativeType);

    void setBuffers(const void* vertices, unsigned int verticesSize, const unsigned int* indices = nullptr, unsigned int indicesSize = 0);
    void setBuffers(unsigned int verticesSize, unsigned int indicesSize);

    void resizeBuffers(unsigned int verticesSize, unsigned int indicesSize);
    void updateAABB(const void* vertices, unsigned int verticesSize);

    EntityType getType() const override { return EntityType::MESH; }

protected:
    GLuint vertexArrayBuffer;

    void setArrayBufferAttributes(const VertexInputAttributes &attributes, unsigned int vertexSize);

    void setMaterialCameraParams(const Camera &camera, const Material* material);
};

#endif // MESH_H
