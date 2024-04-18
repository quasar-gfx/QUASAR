#include <Scene.h>

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

void Scene::bindPBREnvMap(Shader &shader) {
    if (!hasPBREnvMap) {
        return;
    }

    shader.setInt("irradianceMap", Mesh::numTextures + 0);
    irradianceCubeMap.bind(Mesh::numTextures + 0);
    shader.setInt("prefilterMap", Mesh::numTextures + 1);
    prefilterCubeMap.bind(Mesh::numTextures + 1);
    shader.setInt("brdfLUT", Mesh::numTextures + 2);
    brdfLUT.bind(Mesh::numTextures + 2);
}

void Scene::equirectToCubeMap(CubeMap &envCubeMap, Texture &hdrTexture) {
    captureFramebuffer.createColorAndDepthBuffers(envCubeMap.width, envCubeMap.height);
    captureRenderBuffer.create(envCubeMap.width, envCubeMap.height, GL_DEPTH_COMPONENT24);

    captureFramebuffer.bind();
    envCubeMap.loadFromEquirectTexture(*equirectToCubeMapShader, hdrTexture);
    captureFramebuffer.unbind();
}

void Scene::setupIBL(CubeMap &envCubeMap) {
    hasPBREnvMap = true;

    glDisable(GL_BLEND);

    captureFramebuffer.bind();
    captureRenderBuffer.bind();
    captureRenderBuffer.resize(irradianceCubeMap.width, irradianceCubeMap.height);

    captureFramebuffer.bind();
    irradianceCubeMap.convolve(*convolutionShader, envCubeMap);
    captureFramebuffer.unbind();

    captureFramebuffer.bind();
    prefilterCubeMap.prefilter(*prefilterShader, envCubeMap, captureRenderBuffer);
    captureFramebuffer.unbind();

    brdfLUT = Texture({
        .width = envCubeMap.width,
        .height = envCubeMap.height,
        .internalFormat = GL_RG16F,
        .format = GL_RG,
        .type = GL_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR
    });
    brdfFsQuad.init();

    captureFramebuffer.bind();
    captureRenderBuffer.bind();
    captureRenderBuffer.resize(prefilterCubeMap.width, prefilterCubeMap.height);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, brdfLUT.ID, 0);

    brdfShader->bind();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    brdfFsQuad.draw();
    brdfShader->unbind();

    captureFramebuffer.unbind();

    glEnable(GL_BLEND);
}
