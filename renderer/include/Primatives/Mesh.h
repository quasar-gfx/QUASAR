#ifndef MESH_H
#define MESH_H

#include <glad/glad.h>

#include <string>
#include <vector>

#include <Vertex.h>
#include <Shaders/Shader.h>
#include <Texture.h>
#include <Primatives/Entity.h>
#include <Materials/Material.h>
#include <Scene.h>
#include <Camera.h>

struct MeshCreateParams {
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    Material* material;
    bool wireframe = false;
    bool pointcloud = false;
    bool visible = true;
    float IBL = 1.0;
};

class Mesh : public Entity {
public:
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;

    Material* material;

    bool wireframe = false;
    bool pointcloud = false;
    bool visible = true;
    float IBL = 1.0;

    explicit Mesh() : Entity() {}
    explicit Mesh(const MeshCreateParams &params)
            : vertices(params.vertices), indices(params.indices),
              material(params.material),
              wireframe(params.wireframe), pointcloud(params.pointcloud),
              visible(params.visible),
              IBL(params.IBL),
              Entity() {
        createBuffers();
        updateAABB();
    }

    void bindSceneAndCamera(Scene &scene, Camera &camera, const glm::mat4 &model, Material* overrideMaterial = nullptr) override;
    unsigned int draw(Scene &scene, Camera &camera, const glm::mat4 &model, bool frustumCull, Material* overrideMaterial) override;
    unsigned int draw(Scene &scene, Camera &camera, const glm::mat4 &model, const BoundingSphere &boundingSphere, Material* overrideMaterial) override;
    void setBuffers(const std::vector<Vertex> &vertices, const std::vector<unsigned int> &indices);
    void setBuffers(GLuint vertexBufferSSBO, GLuint indexBufferSSBO);
    void updateBuffers();
    void updateAABB();

    void cleanup() {
        material->cleanup();
    }

    EntityType getType() override { return EntityType::MESH; }

protected:
    GLuint vertexArrayBuffer;
    GLuint vertexBuffer;
    GLuint indexBuffer;

    void createBuffers();
    void createAttributes();
};
#endif
