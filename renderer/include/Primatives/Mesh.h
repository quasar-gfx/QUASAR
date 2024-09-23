#ifndef MESH_H
#define MESH_H

#include <vector>

#include <Vertex.h>
#include <Buffer.h>
#include <Shaders/Shader.h>
#include <Texture.h>
#include <Primatives/Entity.h>
#include <Materials/Material.h>
#include <Scene.h>
#include <Cameras/PerspectiveCamera.h>
#include <Cameras/VRCamera.h>

struct MeshDataCreateParams {
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    Material* material;
    bool pointcloud = false;
    float pointSize = 5.0;
    float IBL = 1.0;
    GLenum usage = GL_STATIC_DRAW;
};

struct MeshSizeCreateParams {
    unsigned int numVertices = 0;
    unsigned int numIndices = 0;
    Material* material;
    bool pointcloud = false;
    float pointSize = 5.0;
    float IBL = 1.0;
    GLenum usage = GL_STATIC_DRAW;
};

class Mesh : public Entity {
public:
    Buffer<Vertex> vertexBuffer;
    Buffer<unsigned int> indexBuffer;

    Material* material;

    bool pointcloud = false;
    float pointSize = 5.0;
    float IBL = 1.0;

    GLenum usage;

    Mesh() : vertexBuffer(GL_ARRAY_BUFFER), indexBuffer(GL_ELEMENT_ARRAY_BUFFER) {
        createArrayBuffer();
    }
    Mesh(const MeshDataCreateParams &params)
            : material(params.material)
            , pointcloud(params.pointcloud)
            , pointSize(params.pointSize)
            , IBL(params.IBL)
            , usage(params.usage)
            , vertexBuffer(GL_ARRAY_BUFFER, params.usage)
            , indexBuffer(GL_ELEMENT_ARRAY_BUFFER, params.usage) {
        createArrayBuffer();
        setBuffers(params.vertices, params.indices);
    }
    Mesh(const MeshSizeCreateParams &params)
            : material(params.material)
            , pointcloud(params.pointcloud)
            , pointSize(params.pointSize)
            , IBL(params.IBL)
            , usage(params.usage)
            , vertexBuffer(GL_ARRAY_BUFFER, params.usage)
            , indexBuffer(GL_ELEMENT_ARRAY_BUFFER, params.usage) {
        createArrayBuffer();
        setBuffers(params.numVertices, params.numIndices);
    }

    virtual void bindMaterial(const Scene &scene, const glm::mat4 &model,
                              const Material* overrideMaterial = nullptr, const Texture* prevDepthMap = nullptr) override;

    virtual RenderStats draw(const Camera &camera, const glm::mat4 &model,
                             bool frustumCull = true, const Material* overrideMaterial = nullptr) override;
    virtual RenderStats draw(const Camera &camera, const glm::mat4 &model,
                             const BoundingSphere &boundingSphere, const Material* overrideMaterial = nullptr) override;
    virtual RenderStats draw();

    void setBuffers(const std::vector<Vertex> &vertices);
    void setBuffers(const std::vector<Vertex> &vertices, const std::vector<unsigned int> &indices);
    void setBuffers(unsigned int numVertices, unsigned int numIndices);

    void resizeBuffers(unsigned int vertexBufferSize, unsigned int indexBufferSize);
    void updateAABB(const std::vector<Vertex> &vertices);

    EntityType getType() const override { return EntityType::MESH; }

protected:
    GLuint vertexArrayBuffer;

    void createArrayBuffer();
    void createAttributes();

    void setMaterialCameraParams(const Camera &camera, const Material* material);
};
#endif
