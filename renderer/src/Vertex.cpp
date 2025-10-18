#include <Vertex.h>

using namespace quasar;

Vertex::Vertex(const glm::vec3& position)
    : position(position)
{}
Vertex::Vertex(const glm::vec3& position, const glm::vec2& texCoord, const glm::vec3& normal)
    : position(position), texCoord(texCoord), normal(normal)
{}
Vertex::Vertex(const glm::vec3& position, const glm::vec3& color, const glm::vec3& normal)
    : position(position), color(color), normal(normal)
{}
Vertex::Vertex(const glm::vec3& position, const glm::vec2& texCoord, const glm::vec3& normal, const glm::vec3& tangent, const glm::vec3& bitangent)
    : position(position), texCoord(texCoord), normal(normal), tangent(tangent), bitangent(bitangent)
{}
Vertex::Vertex(const glm::vec3& position, const glm::vec2& texCoord, const glm::vec3& normal, const glm::vec3& tangent)
    : position(position), texCoord(texCoord), normal(normal), tangent(tangent)
{
    bitangent = glm::cross(normal, tangent);
}
