#ifndef AABB_H
#define AABB_H

#include <iostream>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <Vertex.h>

namespace quasar {

class AABB {
public:
    AABB() = default;
    AABB(const glm::vec3 &min, const glm::vec3 &max) : center((min + max) * 0.5f), extents(max.x - center.x, max.y - center.y, max.z - center.z) {}
    AABB(const glm::vec3& center, float iI, float iJ, float iK) : center(center), extents(iI, iJ, iK) {}

    glm::vec3 getCenter() const {
        return center;
    }

    glm::vec3 getExtents() const {
        return extents;
    }

    void update(const glm::vec3 &min, const glm::vec3 &max) {
        center = (min + max) * 0.5f;
        extents = glm::vec3(max.x - center.x, max.y - center.y, max.z - center.z);
    }

    bool intersects(const AABB &other) const {
        return (std::abs(center.x - other.center.x) < (extents.x + other.extents.x)) &&
               (std::abs(center.y - other.center.y) < (extents.y + other.extents.y)) &&
               (std::abs(center.z - other.center.z) < (extents.z + other.extents.z));
    }

private:
    glm::vec3 center = glm::vec3(0.0f);
    glm::vec3 extents = glm::vec3(0.0f);
};

} // namespace quasar

#endif // AABB_H
