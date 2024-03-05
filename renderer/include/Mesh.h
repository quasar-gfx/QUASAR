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
    std::vector<Vertex>       vertices;
    std::vector<unsigned int> indices;
    std::vector<Texture*>     textures;

    float shininess = 1.0f;

    void draw(Shader &shader) override;

    void cleanup() {
        for (auto &texture : textures) {
            if (texture == nullptr) continue;
            texture->cleanup();
        }
    }

    EntityType getType() override { return ENTITY_MESH; }

    static Mesh* create(std::vector<Vertex> &vertices, std::vector<Texture*> &textures, float shininess = 1.0f) {
        return new Mesh(vertices, textures, shininess);
    }

    static Mesh* create(std::vector<Vertex> &vertices, std::vector<unsigned int> &indices, std::vector<Texture*> &textures, float shininess = 1.0f) {
        return new Mesh(vertices, indices, textures, shininess);
    }

protected:
    Mesh(std::vector<Vertex> &vertices, std::vector<Texture*> &textures, float shininess = 1.0f)
            : vertices(vertices), textures(textures), shininess(shininess), Entity() {
        init();
    }

    Mesh(std::vector<Vertex> &vertices, std::vector<unsigned int> &indices, std::vector<Texture*> &textures, float shininess = 1.0f)
            : vertices(vertices), indices(indices), textures(textures), shininess(shininess), Entity() {
        init();
    }

    ~Mesh() {
        cleanup();
    }

private:
    GLuint VAO, VBO, EBO;

    void init();
};
#endif
