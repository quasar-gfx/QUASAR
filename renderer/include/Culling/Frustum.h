#ifndef FRUSTUM_H
#define FRUSTUM_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <Culling/AABB.h>

struct FPlane {
	glm::vec3 normal = { 0.f, 1.f, 0.f };
	float constant = 0.f;

	FPlane() = default;
	FPlane(const glm::vec3& p1, const glm::vec3& norm) : normal(glm::normalize(norm)), constant(-glm::dot(normal, p1)) {}
    FPlane(const glm::vec3& norm, float dist) : normal(glm::normalize(norm)), constant(dist) {}

	float signedDistance(const glm::vec3& point) const {
		return glm::dot(normal, point) + constant;
	}
};

class Frustum {
public:
    explicit Frustum() = default;

    void setFromCameraParams(const glm::vec3 &position, const glm::vec3 &front, const glm::vec3 &right, const glm::vec3 &up, float zNear, float zFar, float aspect, float fovY) {
        float halfVSide = zFar * tanf(fovY * 0.5f);
        float halfHSide = halfVSide * aspect;
        glm::vec3 frontMultFar = zFar * front;

        planes[0] = { position + zNear * front, front };
        planes[1] = { position + frontMultFar, -front };
        planes[2] = { position, glm::cross(frontMultFar - right * halfHSide, up) };
        planes[3] = { position, glm::cross(up, frontMultFar + right * halfHSide) };
        planes[4] = { position, glm::cross(right, frontMultFar - up * halfVSide) };
        planes[5] = { position, glm::cross(frontMultFar + up * halfVSide, right) };
    }

    bool aabbIsVisible(const AABB &aabb, const glm::mat4 &model) const {
		glm::vec3 center = glm::vec3(model * glm::vec4(aabb.getCenter(), 1.0f));

		glm::vec3 right = glm::vec3(model[0]) * aabb.getExtents().x;
        glm::vec3 up = glm::vec3(model[1]) * aabb.getExtents().y;
        glm::vec3 forward = glm::vec3(model[2]) * aabb.getExtents().z;

		float newIi = std::abs(glm::dot(glm::vec3(1.f, 0.f, 0.f), right)) +
                      std::abs(glm::dot(glm::vec3(1.f, 0.f, 0.f), up)) +
                      std::abs(glm::dot(glm::vec3(1.f, 0.f, 0.f), forward));

		float newIj = std::abs(glm::dot(glm::vec3(0.f, 1.f, 0.f), right)) +
                      std::abs(glm::dot(glm::vec3(0.f, 1.f, 0.f), up)) +
                      std::abs(glm::dot(glm::vec3(0.f, 1.f, 0.f), forward));

		float newIk = std::abs(glm::dot(glm::vec3(0.f, 0.f, 1.f), right)) +
                      std::abs(glm::dot(glm::vec3(0.f, 0.f, 1.f), up)) +
                      std::abs(glm::dot(glm::vec3(0.f, 0.f, 1.f), forward));

		AABB globalAABB = AABB(center, newIi, newIj, newIk);

		for (auto& plane : planes) {
            if (!aabbIsOnOrForwardPlane(globalAABB, plane)) {
                return false;
            }
        }
        return true;
    };

    bool aabbIsOnOrForwardPlane(const AABB &aabb, const FPlane &plane) const {
        auto extents = aabb.getExtents();
        auto center = aabb.getCenter();

		const float r = extents.x * std::abs(plane.normal.x) + extents.y * std::abs(plane.normal.y) + extents.z * std::abs(plane.normal.z);
		return -r <= plane.signedDistance(center);
    }

private:
	std::vector<FPlane> planes = std::vector<FPlane>(6);
};

#endif // FRUSTUM_H
