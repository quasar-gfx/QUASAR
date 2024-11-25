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
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    Material* material;
    float IBL = 1.0;
    GLenum usage = GL_STATIC_DRAW;
    bool indirectDraw = false;
};

struct MeshSizeCreateParams {
    unsigned int numVertices = 0;
    unsigned int numIndices = 0;
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
    Buffer<Vertex> vertexBuffer;
    Buffer<unsigned int> indexBuffer;
    Buffer<DrawElementsIndirectCommand> indirectBuffer;

    Material* material;

    float IBL = 1.0;

    GLenum usage;

    bool indirectDraw = false;

    Mesh() : vertexBuffer(GL_ARRAY_BUFFER), indexBuffer(GL_ELEMENT_ARRAY_BUFFER) {
        createArrayBuffer();
    }
    Mesh(const MeshDataCreateParams &params)
            : material(params.material)
            , IBL(params.IBL)
            , usage(params.usage)
            , vertexBuffer(GL_ARRAY_BUFFER, params.usage)
            , indexBuffer(GL_ELEMENT_ARRAY_BUFFER, params.usage)
            , indirectDraw(params.indirectDraw)
            , indirectBuffer(GL_DRAW_INDIRECT_BUFFER, params.usage) {
        createArrayBuffer();
        setBuffers(params.vertices, params.indices);

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
            , vertexBuffer(GL_ARRAY_BUFFER, params.usage)
            , indexBuffer(GL_ELEMENT_ARRAY_BUFFER, params.usage)
            , indirectDraw(params.indirectDraw)
            , indirectBuffer(GL_DRAW_INDIRECT_BUFFER, params.usage) {
        createArrayBuffer();
        setBuffers(params.numVertices, params.numIndices);

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

    void setBuffers(const std::vector<Vertex> &vertices);
    void setBuffers(const std::vector<Vertex> &vertices, const std::vector<unsigned int> &indices);
    void setBuffers(unsigned int numVertices, unsigned int numIndices);

    void resizeBuffers(unsigned int numVertices, unsigned int numIndices);
    void updateAABB(const std::vector<Vertex> &vertices);

    EntityType getType() const override { return EntityType::MESH; }

protected:
    GLuint vertexArrayBuffer;

    void createArrayBuffer();
    void createAttributes();

    void setMaterialCameraParams(const Camera &camera, const Material* material);
};

#endif // MESH_H
