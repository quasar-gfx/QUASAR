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
#include <VertexBuffer.h>
#include <IndexBuffer.h>
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

    void draw(Shader &shader) override;

    void cleanup() {
        vbo.cleanup();
        if (indices.size() > 0) {
            ibo.cleanup();
        }
        for (auto &texture : textures) {
            texture->cleanup();
        }
    }

    EntityType getType() override { return ENTITY_MESH; }

    static Mesh* create(std::vector<Vertex> &vertices, std::vector<Texture*> &textures) {
        return new Mesh(vertices, textures);
    }

    static Mesh* create(std::vector<Vertex> &vertices, std::vector<unsigned int> &indices, std::vector<Texture*> &textures) {
        return new Mesh(vertices, indices, textures);
    }

private:
    VertexBuffer vbo;
    IndexBuffer ibo;

    Mesh(std::vector<Vertex> &vertices, std::vector<Texture*> &textures)
            : vertices(vertices), textures(textures),
            vbo(vertices.data(), static_cast<unsigned int>(vertices.size() * sizeof(Vertex))), Entity() {
        initAttributes();
    }

    Mesh(std::vector<Vertex> &vertices, std::vector<unsigned int> &indices, std::vector<Texture*> &textures)
            : vertices(vertices), indices(indices), textures(textures),
            vbo(vertices.data(), static_cast<unsigned int>(vertices.size() * sizeof(Vertex))),
            ibo(indices.data(), static_cast<unsigned int>(indices.size())), Entity() {
        initAttributes();
    }

    ~Mesh() {
        cleanup();
    }

    void initAttributes() {
        vbo.bind();
        vbo.addAttribute(ATTRIBUTE_POSITION,   3, GL_FALSE, sizeof(Vertex), 0);
        vbo.addAttribute(ATTRIBUTE_NORMAL,     3, GL_FALSE, sizeof(Vertex), offsetof(Vertex, normal));
        vbo.addAttribute(ATTRIBUTE_TEX_COORDS, 2, GL_FALSE, sizeof(Vertex), offsetof(Vertex, texCoords));
        vbo.addAttribute(ATTRIBUTE_TANGENT,    3, GL_FALSE, sizeof(Vertex), offsetof(Vertex, tangent));
        vbo.unbind();
    }
};
#endif
