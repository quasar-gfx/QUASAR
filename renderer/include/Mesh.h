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

class Mesh : public Entity {
public:
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    std::vector<TextureID> textures;

    float shininess = 1.0f;

    Mesh(std::vector<Vertex> &vertices, std::vector<TextureID> &textures, float shininess = 1.0f)
            : vertices(vertices), textures(textures), shininess(shininess), Entity() {
        init();
    }

    Mesh(std::vector<Vertex> &vertices, std::vector<unsigned int> &indices, std::vector<TextureID> &textures, float shininess = 1.0f)
            : vertices(vertices), indices(indices), textures(textures), shininess(shininess), Entity() {
        init();
    }

    void draw(Shader &shader) override;

    void cleanup() {
        for (auto &textureID : textures) {
            if (textureID == 0) continue;
            glDeleteTextures(1, &textureID);
        }
    }

    EntityType getType() override { return ENTITY_MESH; }

private:
    TextureID VAO, VBO, EBO;

    void init();
};
#endif
