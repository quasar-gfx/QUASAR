#include <Scene.h>

using namespace quasar;

Scene::Scene()
    : irradianceCubeMap({
        .width = 32,
        .height = 32,
        .type = CubeMapType::STANDARD,
    })
    , prefilterCubeMap({
        .width = 128,
        .height = 128,
        .type = CubeMapType::PREFILTER,
    })
    , captureRenderTarget({
        .width = 1024,
        .height = 1024,
        .internalFormat = GL_RGB16F,
        .format = GL_RGB,
        .type = GL_HALF_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR ,})
    , captureRenderBuffer({
        .width = 1024,
        .height = 1024,
    })
    , brdfLUT({
        .internalFormat = GL_RG16F,
        .format = GL_RG,
        .type = GL_HALF_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR,
        .data = nullptr,
    })
    , equirectToCubeMapShader({
        .vertexCodeData = SHADER_BUILTIN_CUBEMAP_VERT,
        .vertexCodeSize = SHADER_BUILTIN_CUBEMAP_VERT_len,
        .fragmentCodeData = SHADER_BUILTIN_EQUIRECTANGULAR2CUBEMAP_FRAG,
        .fragmentCodeSize = SHADER_BUILTIN_EQUIRECTANGULAR2CUBEMAP_FRAG_len,
    })
    , convolutionShader({
        .vertexCodeData = SHADER_BUILTIN_CUBEMAP_VERT,
        .vertexCodeSize = SHADER_BUILTIN_CUBEMAP_VERT_len,
        .fragmentCodeData = SHADER_BUILTIN_IRRADIANCE_CONVOLUTION_FRAG,
        .fragmentCodeSize = SHADER_BUILTIN_IRRADIANCE_CONVOLUTION_FRAG_len,
    })
    , prefilterShader({
        .vertexCodeData = SHADER_BUILTIN_CUBEMAP_VERT,
        .vertexCodeSize = SHADER_BUILTIN_CUBEMAP_VERT_len,
        .fragmentCodeData = SHADER_BUILTIN_PREFILTER_FRAG,
        .fragmentCodeSize = SHADER_BUILTIN_PREFILTER_FRAG_len,
    })
    , brdfShader({
        .vertexCodeData = SHADER_BUILTIN_BRDF_VERT,
        .vertexCodeSize = SHADER_BUILTIN_BRDF_VERT_len,
        .fragmentCodeData = SHADER_BUILTIN_BRDF_FRAG,
        .fragmentCodeSize = SHADER_BUILTIN_BRDF_FRAG_len,
    })
    , brdfFsQuad()
    , pointLightUBO({
        .target = GL_UNIFORM_BUFFER,
        .dataSize = sizeof(GPUPointLightBlock),
        .numElems = 1,
        .usage = GL_DYNAMIC_DRAW,
    })
{}

void Scene::addChildNode(Node* node) {
    rootNode.addChildNode(node);
}

void Scene::updateAnimations(float dt) {
    rootNode.updateAnimations(dt);
}

Node* Scene::findNodeByName(const std::string& name) {
    return rootNode.findNodeByName(name);
}

void Scene::setEnvMap(CubeMap* envCubeMap) {
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

int Scene::bindMaterial(const Material* material) {
    auto* shader = material->getShader();

    int texIdx = material->getTextureCount();
    if (hasPBREnvMap) {
        shader->setTexture("material.irradianceMap", irradianceCubeMap, texIdx + 0);
        shader->setTexture("material.prefilterMap", prefilterCubeMap, texIdx + 1);
        shader->setTexture("material.brdfLUT", brdfLUT, texIdx + 2);
    }
    else {
        shader->clearTexture("material.irradianceMap", texIdx + 0);
        shader->clearTexture("material.prefilterMap", texIdx + 1);
        shader->clearTexture("material.brdfLUT", texIdx + 2);
    }
    texIdx += Scene::numTextures;

    texIdx = bindAmbientLight(material, texIdx);
    texIdx = bindDirectionalLight(material, texIdx);
    texIdx = bindPointLights(material, texIdx);

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
                                          directionalLight->shadowMapRenderTarget.depthBuffer, texIdx);
    }
    else {
        material->getShader()->clearTexture("dirLightShadowMap", texIdx);
    }
    texIdx++;

    return texIdx;
}

int Scene::bindPointLights(const Material* material, int texIdx) {
    GPUPointLightBlock uboData{};
    uboData.numPointLights = static_cast<int>(pointLights.size());

    auto* shader = material->getShader();

    for (int i = 0; i < uboData.numPointLights && i < PointLight::maxPointLights; ++i) {
        auto& pointLight = pointLights[i];
        pointLight->setChannel(i);
        uboData.lights[i] = pointLight->toGPULight();

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
    for (int i = uboData.numPointLights; i < PointLight::maxPointLights; ++i) {
        uboData.lights[i] = PointLight::GPUPointLight();
#ifdef GL_CORE
        shader->clearTexture("pointLightShadowMaps[" + std::to_string(i) + "]", texIdx);
#else
        shader->clearTexture("pointLightShadowMaps" + std::to_string(i), texIdx);
#endif
    }

    pointLightUBO.bind();
    pointLightUBO.setData(1, &uboData);
    pointLightUBO.bindToUniformBlock(shader->ID, "PointLightBlock", 0);

    return texIdx;
}

void Scene::equirectToCubeMap(const CubeMap& envCubeMap, const Texture& hdrTexture) {
    captureRenderTarget.bind();
    captureRenderTarget.resize(envCubeMap.width, envCubeMap.height);
    envCubeMap.loadFromEquirectTexture(equirectToCubeMapShader, hdrTexture);
    captureRenderTarget.unbind();
}

void Scene::setupIBL(const CubeMap& envCubeMap) {
    hasPBREnvMap = true;

    captureRenderTarget.resize(envCubeMap.width, envCubeMap.height);

    glDisable(GL_BLEND);

    captureRenderTarget.bind();
    captureRenderBuffer.bind();
    captureRenderBuffer.resize(irradianceCubeMap.width, irradianceCubeMap.height);

    captureRenderTarget.bind();
    irradianceCubeMap.convolve(convolutionShader, envCubeMap);
    captureRenderTarget.unbind();

    captureRenderTarget.bind();
    prefilterCubeMap.prefilter(prefilterShader, envCubeMap, captureRenderBuffer);
    captureRenderTarget.unbind();

    brdfLUT.resize(envCubeMap.width, envCubeMap.height);

    captureRenderTarget.bind();
    captureRenderBuffer.bind();
    captureRenderBuffer.resize(brdfLUT.width, brdfLUT.height);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, brdfLUT.ID, 0);

    brdfShader.bind();
    glViewport(0, 0, brdfLUT.width, brdfLUT.height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    brdfFsQuad.draw();

    captureRenderTarget.unbind();
    captureRenderBuffer.unbind();

    glEnable(GL_BLEND);
}

void Scene::clear() {
    pointLights.clear();
    envCubeMap = nullptr;
    ambientLight = nullptr;
    directionalLight = nullptr;
    hasPBREnvMap = false;
}
