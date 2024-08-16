#ifndef FRUSTUM_H
#define FRUSTUM_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <Culling/AABB.h>

class Frustum {
public:
    struct FPlane {
    public:
        glm::vec3 normal = { 0.0f, 1.0f, 0.0f };
        float constant = 0.0f;

        FPlane() = default;
        FPlane(const glm::vec3 &p1, const glm::vec3 &norm) : normal(glm::normalize(norm)), constant(-glm::dot(normal, p1)) {}
        FPlane(const glm::vec3 &norm, float constant) : normal(norm), constant(constant) {
            float length = glm::length(normal);
            normal /= length;
            constant /= length;
        }

        float signedDistance(const glm::vec3 &point) const {
            return glm::dot(normal, point) + constant;
        }
    };

    Frustum() = default;

    void setFromCameraMatrices(const glm::mat4 &view, const glm::mat4 &projection) {
        glm::mat4 viewProjectionMatrix = projection * view;

        planes[LEFT].normal.x = viewProjectionMatrix[0][3] + viewProjectionMatrix[0][0];
        planes[LEFT].normal.y = viewProjectionMatrix[1][3] + viewProjectionMatrix[1][0];
        planes[LEFT].normal.z = viewProjectionMatrix[2][3] + viewProjectionMatrix[2][0];
        planes[LEFT].constant = viewProjectionMatrix[3][3] + viewProjectionMatrix[3][0];

        planes[RIGHT].normal.x = viewProjectionMatrix[0][3] - viewProjectionMatrix[0][0];
        planes[RIGHT].normal.y = viewProjectionMatrix[1][3] - viewProjectionMatrix[1][0];
        planes[RIGHT].normal.z = viewProjectionMatrix[2][3] - viewProjectionMatrix[2][0];
        planes[RIGHT].constant = viewProjectionMatrix[3][3] - viewProjectionMatrix[3][0];

        planes[BOTTOM].normal.x = viewProjectionMatrix[0][3] + viewProjectionMatrix[0][1];
        planes[BOTTOM].normal.y = viewProjectionMatrix[1][3] + viewProjectionMatrix[1][1];
        planes[BOTTOM].normal.z = viewProjectionMatrix[2][3] + viewProjectionMatrix[2][1];
        planes[BOTTOM].constant = viewProjectionMatrix[3][3] + viewProjectionMatrix[3][1];

        planes[TOP].normal.x = viewProjectionMatrix[0][3] - viewProjectionMatrix[0][1];
        planes[TOP].normal.y = viewProjectionMatrix[1][3] - viewProjectionMatrix[1][1];
        planes[TOP].normal.z = viewProjectionMatrix[2][3] - viewProjectionMatrix[2][1];
        planes[TOP].constant = viewProjectionMatrix[3][3] - viewProjectionMatrix[3][1];

        planes[NEAR].normal.x = viewProjectionMatrix[0][3] + viewProjectionMatrix[0][2];
        planes[NEAR].normal.y = viewProjectionMatrix[1][3] + viewProjectionMatrix[1][2];
        planes[NEAR].normal.z = viewProjectionMatrix[2][3] + viewProjectionMatrix[2][2];
        planes[NEAR].constant = viewProjectionMatrix[3][3] + viewProjectionMatrix[3][2];

        planes[FAR].normal.x = viewProjectionMatrix[0][3] - viewProjectionMatrix[0][2];
        planes[FAR].normal.y = viewProjectionMatrix[1][3] - viewProjectionMatrix[1][2];
        planes[FAR].normal.z = viewProjectionMatrix[2][3] - viewProjectionMatrix[2][2];
        planes[FAR].constant = viewProjectionMatrix[3][3] - viewProjectionMatrix[3][2];

        for (auto &plane : planes) {
            float length = glm::length(plane.normal);
            plane.normal /= length;
            plane.constant /= length;
        }
    }

    void setFromCameraParams(const glm::vec3 &position, const glm::vec3 &front, const glm::vec3 &right, const glm::vec3 &up, float zNear, float zFar, float aspect, float fovY) {
        float halfVSide = zFar * tanf(fovY * 0.5f);
        float halfHSide = halfVSide * aspect;
        glm::vec3 frontMultFar = zFar * front;

        planes[LEFT]   = FPlane(position, glm::cross(up, frontMultFar + right * halfHSide));
        planes[RIGHT]  = FPlane(position, glm::cross(frontMultFar - right * halfHSide, up));
        planes[BOTTOM] = FPlane(position, glm::cross(frontMultFar + up * halfVSide, right));
        planes[TOP]    = FPlane(position, glm::cross(right, frontMultFar - up * halfVSide));
        planes[NEAR]   = FPlane(position + zNear * front, front);
        planes[FAR]    = FPlane(position + frontMultFar, -front);
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

		for (auto &plane : planes) {
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
    enum {
        LEFT = 0,
        RIGHT,
        BOTTOM,
        TOP,
        NEAR,
        FAR,
        COUNT
    };

	std::vector<FPlane> planes = std::vector<FPlane>(COUNT);
};

#endif // FRUSTUM_H
