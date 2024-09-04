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

struct MeshCreateParams {
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    Material* material;
    bool wireframe = false;
    bool pointcloud = false;
    float pointSize = 5.0;
    float IBL = 1.0;
};

class Mesh : public Entity {
public:
    Buffer vertexBuffer;
    Buffer indexBuffer;

    Material* material;

    bool wireframe = false;
    bool pointcloud = false;
    float pointSize = 5.0;
    float IBL = 1.0;

    Mesh() = default;
    Mesh(const MeshCreateParams &params)
            : material(params.material)
            , wireframe(params.wireframe)
            , pointcloud(params.pointcloud)
            , pointSize(params.pointSize)
            , IBL(params.IBL)
            , vertexBuffer(GL_ARRAY_BUFFER)
            , indexBuffer(GL_ELEMENT_ARRAY_BUFFER) {
        createArrayBuffer();
        setBuffers(params.vertices, params.indices);
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
