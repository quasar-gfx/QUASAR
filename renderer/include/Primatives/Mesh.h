#ifndef MESH_H
#define MESH_H

#include <vector>

#include <Vertex.h>
#include <Shaders/Shader.h>
#include <Texture.h>
#include <Primatives/Entity.h>
#include <Materials/Material.h>
#include <Scene.h>
#include <Camera.h>
#include <VRCamera.h>

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
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;

    Material* material;

    bool wireframe = false;
    bool pointcloud = false;
    float pointSize = 5.0;
    float IBL = 1.0;

    explicit Mesh() : Entity() {}
    explicit Mesh(const MeshCreateParams &params)
            : vertices(params.vertices)
            , indices(params.indices)
            , material(params.material)
            , wireframe(params.wireframe)
            , pointcloud(params.pointcloud)
            , pointSize(params.pointSize)
            , IBL(params.IBL)
            , Entity() {
        createBuffers();
        updateAABB();
    }

    virtual void bindMaterial(const Scene &scene, const glm::mat4 &model,
                              const Material* overrideMaterial = nullptr, const Texture* prevDepthMap = nullptr) override;

    virtual RenderStats draw(const Camera &camera, const glm::mat4 &model,
                             bool frustumCull = true, const Material* overrideMaterial = nullptr) override;
    virtual RenderStats draw(const Camera &camera, const glm::mat4 &model,
                             const BoundingSphere &boundingSphere, const Material* overrideMaterial = nullptr) override;
    virtual RenderStats draw(const Camera cameras[], const glm::mat4 &model,
                             bool frustumCull = true, const Material* overrideMaterial = nullptr) override;
    virtual RenderStats draw(const Camera cameras[], const glm::mat4 &model,
                             const BoundingSphere &boundingSphere, const Material* overrideMaterial = nullptr) override;
    virtual RenderStats draw(const VRCamera cameras, const glm::mat4 &model,
                             bool frustumCull = true, const Material* overrideMaterial = nullptr) override;
    virtual RenderStats draw(const VRCamera cameras, const glm::mat4 &model,
                             const BoundingSphere &boundingSphere, const Material* overrideMaterial = nullptr) override;
    virtual RenderStats draw();
    void setBuffers(const std::vector<Vertex> &vertices, const std::vector<unsigned int> &indices);
    void setBuffers(GLuint vertexBufferSSBO, GLuint indexBufferSSBO = -1);
    void updateBuffers();
    void updateAABB();

    EntityType getType() const override { return EntityType::MESH; }

protected:
    GLuint vertexArrayBuffer;
    GLuint vertexBuffer;
    GLuint indexBuffer;

    void createBuffers();
    void createAttributes();

    void setMaterialCameraParams(const Camera &camera, const Material* material);
    void setMaterialCameraParams(const Camera cameras[], const Material* material);
    void setMaterialCameraParams(const VRCamera camera, const Material* material);
};
#endif
