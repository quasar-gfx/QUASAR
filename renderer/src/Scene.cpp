#include <Scene.h>

Scene::Scene() {
    ShaderCreateParams equirectToCubeMapShaderParams = {
        .vertexCodeData = SHADER_CUBEMAP_VERT,
        .vertexCodeSize = SHADER_CUBEMAP_VERT_len,
        .fragmentCodeData = SHADER_EQUIRECTANGULAR2CUBEMAP_FRAG,
        .fragmentCodeSize = SHADER_EQUIRECTANGULAR2CUBEMAP_FRAG_len
    };
    equirectToCubeMapShader = std::make_shared<Shader>(equirectToCubeMapShaderParams);

    ShaderCreateParams convolutionShaderParams = {
        .vertexCodeData = SHADER_CUBEMAP_VERT,
        .vertexCodeSize = SHADER_CUBEMAP_VERT_len,
        .fragmentCodeData = SHADER_IRRADIANCECONVOLUTION_FRAG,
        .fragmentCodeSize = SHADER_IRRADIANCECONVOLUTION_FRAG_len
    };
    convolutionShader = std::make_shared<Shader>(convolutionShaderParams);

    ShaderCreateParams prefilterShaderParams = {
        .vertexCodeData = SHADER_CUBEMAP_VERT,
        .vertexCodeSize = SHADER_CUBEMAP_VERT_len,
        .fragmentCodeData = SHADER_PREFILTER_FRAG,
        .fragmentCodeSize = SHADER_PREFILTER_FRAG_len
    };
    prefilterShader = std::make_shared<Shader>(prefilterShaderParams);

    ShaderCreateParams brdfShaderParams = {
        .vertexCodeData = SHADER_BRDF_VERT,
        .vertexCodeSize = SHADER_BRDF_VERT_len,
        .fragmentCodeData = SHADER_BRDF_FRAG,
        .fragmentCodeSize = SHADER_BRDF_FRAG_len
    };
    brdfShader = std::make_shared<Shader>(brdfShaderParams);
}

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

void Scene::bindMaterial(Material* material) {
    if (!hasPBREnvMap) {
        return;
    }

    material->shader->setInt("irradianceMap", material->getTextureCount() + 0);
    irradianceCubeMap.bind(material->getTextureCount() + 0);
    material->shader->setInt("prefilterMap", material->getTextureCount() + 1);
    prefilterCubeMap.bind(material->getTextureCount() + 1);
    material->shader->setInt("brdfLUT", material->getTextureCount() + 2);
    brdfLUT.bind(material->getTextureCount() + 2);
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
    captureRenderBuffer.resize(brdfLUT.width, brdfLUT.height);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, brdfLUT.ID, 0);

    brdfShader->bind();
    glViewport(0, 0, brdfLUT.width, brdfLUT.height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    brdfFsQuad.draw();
    brdfShader->unbind();

    captureFramebuffer.unbind();
    captureRenderBuffer.unbind();

    glEnable(GL_BLEND);
}
