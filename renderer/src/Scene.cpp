#include <Scene.h>

using namespace quasar;

void Scene::setEnvMap(SkyBox* envCubeMap) {
    this->envCubeMap = envCubeMap;
}

void Scene::setAmbientLight(AmbientLight* ambientLight) {
    this->ambientLight = ambientLight;
}

void Scene::setDirectionalLight(DirectionalLight* directionalLight) {
    this->directionalLight = directionalLight;
}

void Scene::addPointLight(PointLight* pointLight) {
    if (pointLights.size() >= PointLight::maxPointLights) {
        spdlog::warn("Maximum number of point lights reached, cannot add more.");
        return;
    }
    pointLights.push_back(pointLight);
}

int Scene::bindMaterial(const Material* material, Buffer& pointLightsUBO) {
    auto shader = material->getShader();

    int texIdx = material->getTextureCount();
    if (envCubeMap != nullptr) {
        shader->setTexture("material.irradianceMap", envCubeMap->getIrradianceCubeMap(), texIdx + 0);
        shader->setTexture("material.prefilterMap", envCubeMap->getPrefilterCubeMap(), texIdx + 1);
        shader->setTexture("material.brdfLUT", envCubeMap->getBRDFLUT(), texIdx + 2);
    }
    else {
        shader->clearTexture("material.irradianceMap", texIdx + 0);
        shader->clearTexture("material.prefilterMap", texIdx + 1);
        shader->clearTexture("material.brdfLUT", texIdx + 2);
    }
    texIdx += Scene::numTextures;

    texIdx = bindAmbientLight(material, texIdx);
    texIdx = bindDirectionalLight(material, texIdx);
    texIdx = bindPointLights(material, pointLightsUBO, texIdx);

    return texIdx;
}

int Scene::bindAmbientLight(const Material* material, int texIdx) {
    if (ambientLight != nullptr) {
        ambientLight->bindMaterial(material);
    }
    return texIdx;
}

int Scene::bindDirectionalLight(const Material* material, int texIdx) {
    if (directionalLight != nullptr) {
        directionalLight->bindMaterial(material);
        material->getShader()->setMat4("lightSpaceMatrix", directionalLight->lightSpaceMatrix);

        // Bind shadow map
        material->getShader()->setTexture("dirLightShadowMap",
                                          directionalLight->shadowMapRenderTarget.depthTexture, texIdx);
    }
    else {
        material->getShader()->clearTexture("dirLightShadowMap", texIdx);
    }
    texIdx++;

    return texIdx;
}

int Scene::bindPointLights(const Material* material, Buffer& pointLightsUBO, int texIdx) {
    pointLightsData.numPointLights = static_cast<int>(pointLights.size());

    auto shader = material->getShader();

    for (int i = 0; i < pointLightsData.numPointLights && i < PointLight::maxPointLights; i++) {
        auto& pointLight = pointLights[i];
        pointLight->setChannel(i);
        pointLightsData.lights[i] = pointLight->toGPULight();

        // Bind shadow maps
#ifdef GL_CORE
        shader->setTexture("pointLightShadowMaps[" + std::to_string(i) + "]",
                           pointLight->shadowMapRenderTarget.depthCubeMap, texIdx);
#else
        // This is a hack to get around GLES's lack of samplerCube arrays
        shader->setTexture("pointLightShadowMaps" + std::to_string(i),
                           pointLight->shadowMapRenderTarget.depthCubeMap, texIdx);
#endif
        texIdx++;
    }
    // Set empty lights and shadow maps for remaining slots
    for (int i = pointLightsData.numPointLights; i < PointLight::maxPointLights; i++) {
        pointLightsData.lights[i] = PointLight::GPUPointLight();
#ifdef GL_CORE
        shader->clearTexture("pointLightShadowMaps[" + std::to_string(i) + "]", texIdx);
#else
        shader->clearTexture("pointLightShadowMaps" + std::to_string(i), texIdx);
#endif
    }

    pointLightsUBO.bind();
    pointLightsUBO.setData(1, &pointLightsData);
    pointLightsUBO.bindToUniformBlock(shader->ID, "PointLightBlock", 0);

    return texIdx;
}

void Scene::clear() {
    pointLights.clear();
    envCubeMap = nullptr;
    ambientLight = nullptr;
    directionalLight = nullptr;
}
