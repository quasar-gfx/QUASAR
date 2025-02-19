#include <Materials/DeferredLightingMaterial.h>

Shader* DeferredLightingMaterial::shader = nullptr;

DeferredLightingMaterial::DeferredLightingMaterial() {
    if (shader == nullptr) {
        ShaderDataCreateParams dirShadowMapParams{
            .vertexCodeData = SHADER_BUILTIN_POSTPROCESS_VERT,
            .vertexCodeSize = SHADER_BUILTIN_POSTPROCESS_VERT_len,
            .fragmentCodeData = SHADER_BUILTIN_DEFERRED_LIGHTING_FRAG,
            .fragmentCodeSize = SHADER_BUILTIN_DEFERRED_LIGHTING_FRAG_len
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

void DeferredLightingMaterial::bindGBuffer(const DeferredGBuffer &gBuffer) const {
    shader->setTexture("gAlbedo", gBuffer.albedoBuffer, 0);
    shader->setTexture("gPBR", gBuffer.pbrBuffer, 1);
    shader->setTexture("gEmissive", gBuffer.emissiveBuffer, 2);
    shader->setTexture("gLightPositionXYZ", gBuffer.lightPositionXYZBuffer, 3);
    shader->setTexture("gLightPositionWIBLAlpha", gBuffer.lightPositionWIBLAlphaBuffer, 4);
    shader->setTexture("gPosition", gBuffer.positionBuffer, 5);
    shader->setTexture("gNormal", gBuffer.normalsBuffer, 6);
}

void DeferredLightingMaterial::bindCamera(const Camera &camera) const {
    shader->setVec3("camera.position", camera.getPosition());
}
