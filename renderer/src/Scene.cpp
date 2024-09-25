#include <Scene.h>

Scene::Scene()
        : irradianceCubeMap({ .width = 32, .height = 32, .type = CubeMapType::STANDARD })
        , prefilterCubeMap({ .width = 128, .height = 128, .type = CubeMapType::PREFILTER })
        , captureRenderTarget({ .width = 512,
                                .height = 512,
                                .internalFormat = GL_RGB16F,
                                .format = GL_RGB,
                                .type = GL_FLOAT,
                                .wrapS = GL_CLAMP_TO_EDGE,
                                .wrapT = GL_CLAMP_TO_EDGE,
                                .minFilter = GL_LINEAR,
                                .magFilter = GL_LINEAR })
        , captureRenderBuffer({ .width = 512, .height = 512 })
        , brdfLUT({
            .internalFormat = GL_RG16F,
            .format = GL_RG,
            .type = GL_FLOAT,
            .wrapS = GL_CLAMP_TO_EDGE,
            .wrapT = GL_CLAMP_TO_EDGE,
            .minFilter = GL_LINEAR,
            .magFilter = GL_LINEAR,
            .data = nullptr
        })
        , brdfFsQuad()
        , equirectToCubeMapShader({
            .vertexCodeData = SHADER_CUBEMAP_VERT,
            .vertexCodeSize = SHADER_CUBEMAP_VERT_len,
            .fragmentCodeData = SHADER_EQUIRECTANGULAR2CUBEMAP_FRAG,
            .fragmentCodeSize = SHADER_EQUIRECTANGULAR2CUBEMAP_FRAG_len })
        , convolutionShader({
            .vertexCodeData = SHADER_CUBEMAP_VERT,
            .vertexCodeSize = SHADER_CUBEMAP_VERT_len,
            .fragmentCodeData = SHADER_IRRADIANCECONVOLUTION_FRAG,
            .fragmentCodeSize = SHADER_IRRADIANCECONVOLUTION_FRAG_len })
        , prefilterShader({
            .vertexCodeData = SHADER_CUBEMAP_VERT,
            .vertexCodeSize = SHADER_CUBEMAP_VERT_len,
            .fragmentCodeData = SHADER_PREFILTER_FRAG,
            .fragmentCodeSize = SHADER_PREFILTER_FRAG_len })
        , brdfShader({
            .vertexCodeData = SHADER_BRDF_VERT,
            .vertexCodeSize = SHADER_BRDF_VERT_len,
            .fragmentCodeData = SHADER_BRDF_FRAG,
            .fragmentCodeSize = SHADER_BRDF_FRAG_len }) { }

void Scene::addChildNode(Node* node) {
    children.push_back(node);
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
    pointLights.push_back(pointLight);
}

void Scene::bindMaterial(const Material* material) const {
    if (hasPBREnvMap) {
        material->getShader()->setTexture("material.irradianceMap", irradianceCubeMap, material->getTextureCount());
        material->getShader()->setTexture("material.prefilterMap", prefilterCubeMap, material->getTextureCount() + 1);
        material->getShader()->setTexture("material.brdfLUT", brdfLUT, material->getTextureCount() + 2);
    }
}

void Scene::equirectToCubeMap(const CubeMap &envCubeMap, const Texture &hdrTexture) {
    captureRenderTarget.bind();
    captureRenderTarget.resize(envCubeMap.width, envCubeMap.height);
    envCubeMap.loadFromEquirectTexture(equirectToCubeMapShader, hdrTexture);
    captureRenderTarget.unbind();
}

void Scene::setupIBL(const CubeMap &envCubeMap) {
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
    children.clear();
    pointLights.clear();
    envCubeMap = nullptr;
    ambientLight = nullptr;
    directionalLight = nullptr;
    hasPBREnvMap = false;
}
