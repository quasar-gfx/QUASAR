#include <Lights/PointLight.h>
#include <Materials/DeferredLightingMaterial.h>

using namespace quasar;

Shader* DeferredLightingMaterial::shader = nullptr;

DeferredLightingMaterial::DeferredLightingMaterial() {
    if (shader == nullptr) {
        ShaderDataCreateParams dirShadowMapParams{
            .vertexCodeData = SHADER_BUILTIN_POSTPROCESS_VERT,
            .vertexCodeSize = SHADER_BUILTIN_POSTPROCESS_VERT_len,
            .fragmentCodeData = SHADER_BUILTIN_DEFERRED_LIGHTING_FRAG,
            .fragmentCodeSize = SHADER_BUILTIN_DEFERRED_LIGHTING_FRAG_len,
            .defines = {
                "#define MAX_POINT_LIGHTS " + std::to_string(PointLight::maxPointLights),
            }
        };
        shader = new Shader(dirShadowMapParams);
    }
}

DeferredLightingMaterial::~DeferredLightingMaterial() {
    if (shader != nullptr) {
        delete shader;
        shader = nullptr;
    }
}

void DeferredLightingMaterial::bindGBuffer(const GBuffer& gBuffer) const {
    shader->setTexture("gAlbedo", gBuffer.albedoBuffer, 0);
    shader->setTexture("gPBR", gBuffer.pbrBuffer, 1);
    shader->setTexture("gAlpha", gBuffer.alphaBuffer, 2);
    shader->setTexture("gNormal", gBuffer.normalsBuffer, 3);
    shader->setTexture("gPosition", gBuffer.positionBuffer, 4);
    shader->setTexture("gLightPosition", gBuffer.lightPositionBuffer, 5);
}

void DeferredLightingMaterial::bindCamera(const Camera& camera) const {
    shader->setVec3("camera.position", camera.getPosition());
}
