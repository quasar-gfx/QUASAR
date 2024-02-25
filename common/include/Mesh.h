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

enum VertexAttribute {
    ATTRIBUTE_POSITION   = 0,
    ATTRIBUTE_NORMAL     = 1,
    ATTRIBUTE_TEX_COORDS = 2,
    ATTRIBUTE_TANGENT    = 3
};

class Mesh {
public:
    std::vector<Vertex>       vertices;
    std::vector<unsigned int> indices;
    std::vector<Texture*>      textures;

    void draw(Shader &shader) {
        int diffuseIdx  = 1;
        int specularIdx = 1;
        int normalIdx   = 1;
        int heightIdx   = 1;

        for (int i = 0; i < textures.size(); i++) {
            std::string name;

            TextureType type = textures[i]->type;
            if (type == TEXTURE_DIFFUSE) {
                name = "texture_diffuse" + std::to_string(diffuseIdx++);
            }
            else if (type == TEXTURE_SPECULAR) {
                name = "texture_specular" + std::to_string(specularIdx++);
            }
            else if (type == TEXTURE_NORMAL) {
                name = "texture_normal" + std::to_string(normalIdx++);
            }
            else if (type == TEXTURE_HEIGHT) {
                name = "texture_height" + std::to_string(heightIdx++);
            }

            shader.setInt(name.c_str(), i);
            textures[i]->bind(i);
        }

        vbo.bind();
        if (indices.size() > 0) {
            ibo.bind();
            glDrawElements(GL_TRIANGLES, static_cast<unsigned int>(indices.size()), GL_UNSIGNED_INT, 0);
            ibo.unbind();
        }
        else {
            glDrawArrays(GL_TRIANGLES, 0, static_cast<unsigned int>(vertices.size()));
        }
        vbo.unbind();

        if (textures.size() > 0) {
            textures[textures.size()-1]->unbind();
        }
    }

    void cleanup() {
        vbo.cleanup();
        if (indices.size() > 0) {
            ibo.cleanup();
        }
        for (auto &texture : textures) {
            texture->cleanup();
        }
    }

    static Mesh* create(std::vector<Vertex> &vertices, std::vector<Texture*> &textures) {
        return new Mesh(vertices, textures);
    }

    static Mesh* create(std::vector<Vertex> &vertices, std::vector<unsigned int> &indices, std::vector<Texture*> &textures) {
        return new Mesh(vertices, indices, textures);
    }

    static Mesh* create(const std::string& path, std::vector<Texture*>& textures);

private:
    VertexBuffer vbo;
    IndexBuffer ibo;

    Mesh(std::vector<Vertex> &vertices, std::vector<Texture*> &textures)
            : vertices(vertices), textures(textures),
            vbo(vertices.data(), static_cast<unsigned int>(vertices.size() * sizeof(Vertex))) {
        initAttributes();
    }

    Mesh(std::vector<Vertex> &vertices, std::vector<unsigned int> &indices, std::vector<Texture*> &textures)
            : vertices(vertices), indices(indices), textures(textures),
            vbo(vertices.data(), static_cast<unsigned int>(vertices.size() * sizeof(Vertex))),
            ibo(indices.data(), static_cast<unsigned int>(indices.size())) {
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
