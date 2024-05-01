#ifndef MESH_H
#define MESH_H

#include <glad/glad.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#include <string>
#include <vector>

#include <Shaders/Shader.h>
#include <Texture.h>
#include <Primatives/Entity.h>
#include <Materials/Material.h>
#include <Scene.h>
#include <Camera.h>

enum VertexAttribute {
    ATTRIBUTE_POSITION   = 0,
    ATTRIBUTE_NORMAL     = 1,
    ATTRIBUTE_TEX_COORDS = 2,
    ATTRIBUTE_TANGENT    = 3,
    ATTRIBUTE_BITANGENT  = 4
};

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texCoords;
    glm::vec3 tangent;
    glm::vec3 bitangent;

    bool operator==(const Vertex& other) const {
        return position == other.position && normal == other.normal && texCoords == other.texCoords && tangent == other.tangent;
    }
};

namespace std {
    template<> struct hash<Vertex> {
        size_t operator()(Vertex const& vertex) const {
            return ((hash<glm::vec3>()(vertex.position) ^
                   (hash<glm::vec3>()(vertex.normal) << 1)) >> 1) ^
                   (hash<glm::vec2>()(vertex.texCoords) << 1 ^
                   (hash<glm::vec3>()(vertex.tangent) << 1) >> 1);
        }
    };
}

struct MeshCreateParams {
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    Material* material;
    bool wireframe = false;
    bool pointcloud = false;
    bool IBL = true;
    bool transparent = false;
    bool metalRoughnessCombined = false;
};

class Mesh : public Entity {
public:
    Material* material;

    bool wireframe = false;
    bool pointcloud = false;
    bool IBL = true;
    bool transparent = false;
    bool metalRoughnessCombined = false;

    explicit Mesh() : Entity() {}

    explicit Mesh(const MeshCreateParams &params)
            : vertices(params.vertices), indices(params.indices),
                material(params.material),
                wireframe(params.wireframe), pointcloud(params.pointcloud),
                IBL(params.IBL), transparent(params.transparent),
                metalRoughnessCombined(params.metalRoughnessCombined),
                Entity() {
        init();
    }

    void bindSceneAndCamera(Scene &scene, Camera &camera, glm::mat4 model, Material* overrideMaterial = nullptr) override;
    void draw(Material* overrideMaterial) override;

    void cleanup() {
        material->cleanup();
    }

    EntityType getType() override { return EntityType::MESH; }

protected:
    TextureID VAO, VBO, EBO;

    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;

    void init();
};
#endif
