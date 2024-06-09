#ifndef VERTEX_H
#define VERTEX_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

enum VertexAttribute {
    ATTRIBUTE_ID = 0,
    ATTRIBUTE_POSITION,
    ATTRIBUTE_COLOR,
    ATTRIBUTE_NORMAL,
    ATTRIBUTE_TEX_COORDS,
    ATTRIBUTE_TANGENT,
    ATTRIBUTE_BITANGENT
};

struct Vertex {
    uint32_t ID;
    glm::vec3 position;
    glm::vec3 color = glm::vec3(1.0f);
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
    Vertex(glm::vec3 position, glm::vec3 color, glm::vec3 normal)
        : position(position), color(color), normal(normal) {
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

#endif // VERTEX_H
