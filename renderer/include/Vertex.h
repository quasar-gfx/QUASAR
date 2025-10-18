#ifndef VERTEX_H
#define VERTEX_H

#include <vector>

#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#include <Utils/Platform.h>

namespace quasar {

struct VertexInputAttribute {
    GLuint index;
    GLint size;
    GLenum type;
    GLboolean normalized;
    GLintptr pointer;
};
typedef std::vector<VertexInputAttribute> VertexInputAttributes;

struct Vertex {
    alignas(16) glm::vec3 position;
    alignas(16) glm::vec3 color = glm::vec3(1.0f);
    alignas(16) glm::vec2 texCoord;
    alignas(16) glm::vec3 normal;
    alignas(16) glm::vec3 tangent;
    alignas(16) glm::vec3 bitangent;

    Vertex() = default;
    Vertex(const glm::vec3& position);
    Vertex(const glm::vec3& position, const glm::vec2& texCoord, const glm::vec3& normal);
    Vertex(const glm::vec3& position, const glm::vec3& color, const glm::vec3& normal);
    Vertex(const glm::vec3& position, const glm::vec2& texCoord, const glm::vec3& normal, const glm::vec3& tangent, const glm::vec3& bitangent);
    Vertex(const glm::vec3& position, const glm::vec2& texCoord, const glm::vec3& normal, const glm::vec3& tangent);

    bool operator==(const Vertex& other) const {
        return position == other.position && color == other.color && texCoord == other.texCoord &&
               normal == other.normal && tangent == other.tangent && bitangent == other.bitangent;
    }

    static const VertexInputAttributes getVertexInputAttributes() {
        return {
            {0, 3, GL_FLOAT, GL_FALSE, offsetof(Vertex, position)},
            {1, 3, GL_FLOAT, GL_FALSE, offsetof(Vertex, color)},
            {2, 2, GL_FLOAT, GL_FALSE, offsetof(Vertex, texCoord)},
            {3, 3, GL_FLOAT, GL_FALSE, offsetof(Vertex, normal)},
            {4, 3, GL_FLOAT, GL_FALSE, offsetof(Vertex, tangent)},
            {5, 3, GL_FLOAT, GL_FALSE, offsetof(Vertex, bitangent)},
        };
    }
};

} // namespace quasar

namespace std {
    template<> struct hash<quasar::Vertex> {
        size_t operator()(quasar::Vertex const& vertex) const {
            return hash<glm::vec3>()(vertex.position) ^
                   hash<glm::vec3>()(vertex.color) ^
                   hash<glm::vec2>()(vertex.texCoord) ^
                   hash<glm::vec3>()(vertex.normal) ^
                   hash<glm::vec3>()(vertex.tangent) ^
                   hash<glm::vec3>()(vertex.bitangent);
        }
    };
}

#endif // VERTEX_H
