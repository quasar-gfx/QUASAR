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
    ATTRIBUTE_ID = 0,
    ATTRIBUTE_POSITION,
    ATTRIBUTE_NORMAL,
    ATTRIBUTE_TEX_COORDS,
    ATTRIBUTE_TANGENT,
    ATTRIBUTE_BITANGENT
};

struct Vertex {
    uint32_t ID;
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texCoords;
    glm::vec3 tangent;
    glm::vec3 bitangent;

    bool operator==(const Vertex& other) const {
        return position == other.position && normal == other.normal && texCoords == other.texCoords &&
               tangent == other.tangent && bitangent == other.bitangent;
    }

    Vertex() {
        ID = nextID++;
    }
    Vertex(glm::vec3 position, glm::vec3 normal, glm::vec2 texCoords)
        : position(position), normal(normal), texCoords(texCoords) {
        ID = nextID++;
    }
    Vertex(glm::vec3 position, glm::vec3 normal, glm::vec2 texCoords, glm::vec3 tangent, glm::vec3 bitangent)
        : position(position), normal(normal), texCoords(texCoords), tangent(tangent), bitangent(bitangent) {
        ID = nextID++;
    }
    Vertex(glm::vec3 position, glm::vec3 normal, glm::vec2 texCoords, glm::vec3 tangent)
        : position(position), normal(normal), texCoords(texCoords), tangent(tangent) {
        bitangent = glm::cross(normal, tangent);
        ID = nextID++;
    }

    static uint32_t nextID;
};

namespace std {
    template<> struct hash<Vertex> {
        size_t operator()(Vertex const& vertex) const {
            return ((hash<glm::vec3>()(vertex.position) ^
                   (hash<glm::vec3>()(vertex.normal) << 1)) >> 1) ^
                   (hash<glm::vec2>()(vertex.texCoords) << 1 ^
                   (hash<glm::vec3>()(vertex.tangent) << 1) >> 1) ^
                   (hash<glm::vec3>()(vertex.bitangent) << 1 >> 1);
        }
    };
}

struct MeshCreateParams {
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    Material* material;
    bool wireframe = false;
    bool pointcloud = false;
    float IBL = 1.0;
};

class Mesh : public Entity {
public:
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;

    Material* material;

    bool wireframe = false;
    bool pointcloud = false;
    float IBL = 1.0;

    explicit Mesh() : Entity() {}

    explicit Mesh(const MeshCreateParams &params)
            : vertices(params.vertices), indices(params.indices),
              material(params.material),
              wireframe(params.wireframe), pointcloud(params.pointcloud),
              IBL(params.IBL),
              Entity() {
        createBuffers();
    }

    void bindSceneAndCamera(Scene &scene, Camera &camera, glm::mat4 model, Material* overrideMaterial = nullptr) override;
    unsigned int draw(Material* overrideMaterial) override;

    void cleanup() {
        material->cleanup();
    }

    EntityType getType() override { return EntityType::MESH; }

protected:
    TextureID VAO, VBO, EBO;

    void createBuffers();
};
#endif
