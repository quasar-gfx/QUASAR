#ifndef BOUNDING_SPHERE_H
#define BOUNDING_SPHERE_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <Culling/AABB.h>

namespace quasar {

class BoundingSphere {
public:
    BoundingSphere() = default;
    BoundingSphere(const glm::vec3 &center, float radius) : center(center), radius(radius) {}

    glm::vec3 getCenter() const {
        return center;
    }

    float getRadius() const {
        return radius;
    }

    void update(const glm::vec3 &center, float radius) {
        this->center = center;
        this->radius = radius;
    }

    bool intersects(const BoundingSphere &other) const {
        return glm::distance(center, other.center) < (radius + other.radius);
    }

    bool intersects(const glm::mat4 &transform, const AABB &aabb) const {
        auto aabbCenter = glm::vec3(transform * glm::vec4(aabb.getCenter(), 1.0f));
        auto aabbExtents = glm::vec3(transform * glm::vec4(aabb.getExtents(), 0.0f));

        glm::vec3 closestPointInAABB;
        closestPointInAABB.x = glm::clamp(center.x, aabbCenter.x - aabbExtents.x, aabbCenter.x + aabbExtents.x);
        closestPointInAABB.y = glm::clamp(center.y, aabbCenter.y - aabbExtents.y, aabbCenter.y + aabbExtents.y);
        closestPointInAABB.z = glm::clamp(center.z, aabbCenter.z - aabbExtents.z, aabbCenter.z + aabbExtents.z);

        return glm::distance(center, closestPointInAABB) < radius;
    }

private:
    glm::vec3 center = glm::vec3(0.0f);
    float radius = 0.0f;
};

} // namespace quasar

#endif // BOUNDING_SPHERE_H
