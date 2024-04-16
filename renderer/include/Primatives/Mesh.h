#ifndef MESH_H
#define MESH_H

#include <glad/glad.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#include <string>
#include <vector>

#include <Shader.h>
#include <Texture.h>
#include <Entity.h>
#include <Materials/Material.h>

enum VertexAttribute {
    ATTRIBUTE_POSITION   = 0,
    ATTRIBUTE_NORMAL     = 1,
    ATTRIBUTE_TEX_COORDS = 2,
    ATTRIBUTE_TANGENT    = 3
};

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texCoords;
    glm::vec3 tangent;

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
};

class Mesh : public Entity {
public:
    Material* material;

    bool wireframe = false;
    bool pointcloud = false;

    explicit Mesh() : Entity() {}

    explicit Mesh(const MeshCreateParams &params)
            : vertices(params.vertices), indices(params.indices),
                material(params.material),
                wireframe(params.wireframe), pointcloud(params.pointcloud),
                Entity() {
        init();
    }

    void draw(Shader &shader) override;

    void cleanup() {
        material->cleanup();
    }

    EntityType getType() override { return ENTITY_MESH; }

    static const unsigned int numTextures = 5;

protected:
    TextureID VAO, VBO, EBO;

    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;

    void init();
};
#endif
