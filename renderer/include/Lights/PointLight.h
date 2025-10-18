#ifndef POINT_LIGHT_H
#define POINT_LIGHT_H

#include <spdlog/spdlog.h>

#include <Lights/Light.h>
#include <CubeMap.h>
#include <RenderTargets/PointLightShadowRT.h>
#include <Materials/PointShadowMapMaterial.h>
#include <Culling/BoundingSphere.h>

namespace quasar {

struct PointLightCreateParams {
    glm::vec3 color = glm::vec3(1.0f);
    glm::vec3 position = glm::vec3(0.0f);
    float intensity = 1.0f;
    float constant = 1.0f;
    float linear = 0.09f;
    float quadratic = 0.032f;
    float intensityThreshold = 1.0f;
    float shadowNear = 0.1f;
    float shadowFar = 100.0f;
    float shadowFov = 90.0f;
    uint shadowMapRes = 1024;
    bool debug = false;
};

class PointLight : public Light {
public:
#if defined(PLATFORM_GLES) // We limit GLES to 4 point lights due to performance constraints
    static const uint maxPointLights = 4;
#elif defined(__APPLE__) // Apple platforms have a limit on number of texture units
    static const uint maxPointLights = 5;
#else
    static const uint maxPointLights = 8;
#endif

    struct GPUPointLight {
        glm::vec3 position;
        int shadowIndex;

        glm::vec3 color;
        float intensity;

        float constant;
        float linear;
        float quadratic;
        float farPlane;
    };

    glm::vec3 position = glm::vec3(0.0f);
    float constant = 1.0f;
    float linear = 0.09f;
    float quadratic = 0.032f;
    float intensityThreshold = 1.0f;

    bool debug = false;

    int channel = -1;

    glm::mat4 lookAtPerFace[NUM_CUBEMAP_FACES];
    PointLightShadowRT shadowMapRenderTarget;
    PointShadowMapMaterial shadowMapMaterial;

    BoundingSphere boundingSphere;

    PointLight(const PointLightCreateParams& params)
        : position(params.position)
        , constant(params.constant)
        , linear(params.linear)
        , quadratic(params.quadratic)
        , intensityThreshold(params.intensityThreshold)
        , Light({
            .color = params.color,
            .intensity = params.intensity,
            .shadowNear = params.shadowNear,
            .shadowFar = params.shadowFar,
            .shadowMapRes = params.shadowMapRes,
        })
        , shadowMapRenderTarget({
            .width = shadowMapRes,
            .height = shadowMapRes,
        })
        , boundingSphere(position, getLightRadius())
        , debug(params.debug)
    {
        shadowProjectionMat = glm::perspective(glm::radians(params.shadowFov), 1.0f, params.shadowNear, params.shadowFar);
        updateLookAtFace();
    }

    void setChannel(int channel) {
        this->channel = channel;
    }

    void bindMaterial(const Material* material) override {
        if (channel == -1) {
            spdlog::warn("Point light channel is not set!");
            return;
        }
    }

    void setPosition(const glm::vec3& position) {
        this->position = position;
        updateLookAtFace();
        updateBoundingSphere();
    }

    void setAttenuation(float constant, float linear, float quadratic) {
        this->constant = constant;
        this->linear = linear;
        this->quadratic = quadratic;
        updateBoundingSphere();
    }

    void updateBoundingSphere() {
        boundingSphere.update(position, getLightRadius());
    }

    float getLightRadius() {
        float discriminant = linear * linear - 4.0f * quadratic * (constant - intensity / intensityThreshold);
        if (discriminant < 0.0f) {
            return 0.0f; // light does not reach the intensity threshold
        }

        float root1 = (-linear + std::sqrt(discriminant)) / (2.0f * quadratic);
        float root2 = (-linear - std::sqrt(discriminant)) / (2.0f * quadratic);

        if (root1 > 0.0f && root2 > 0.0f) {
            return std::max(root1, root2);
        }
        else if (root1 > 0.0f) {
            return root1;
        }
        else if (root2 > 0.0f) {
            return root2;
        }
        else {
            return 0.0f;
        }
    }

    const GPUPointLight toGPULight() const {
        return {
            .position = position,
            .shadowIndex = channel,
            .color = color,
            .intensity = intensity,
            .constant = constant,
            .linear = linear,
            .quadratic = quadratic,
            .farPlane = shadowFar,
        };
    }

private:
    void updateLookAtFace() {
        lookAtPerFace[0] = glm::lookAt(position, position + glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f));
        lookAtPerFace[1] = glm::lookAt(position, position + glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f));
        lookAtPerFace[2] = glm::lookAt(position, position + glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        lookAtPerFace[3] = glm::lookAt(position, position + glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f));
        lookAtPerFace[4] = glm::lookAt(position, position + glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, -1.0f, 0.0f));
        lookAtPerFace[5] = glm::lookAt(position, position + glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, -1.0f, 0.0f));
    }
};

} // namespace quasar

#endif // POINT_LIGHT_H
