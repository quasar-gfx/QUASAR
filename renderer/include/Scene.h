#ifndef SCENE_H
#define SCENE_H

#include <vector>

#include <Entity.h>
#include <Lights.h>
#include <CubeMap.h>
#include <Mesh.h>
#include <FrameBuffer.h>
#include <FullScreenQuad.h>

class Scene {
public:
    CubeMap* envCubeMap = nullptr;
    AmbientLight* ambientLight = nullptr;
    DirectionalLight* directionalLight = nullptr;
    std::vector<PointLight*> pointLights;

    std::vector<Node*> children;

    bool hasPBREnvMap = false;

    // set up framebuffer
    FrameBuffer captureFramebuffer;
    RenderBuffer captureRenderBuffer;

    // create an irradiance cubemap, and rescale capture FBO to irradiance scale
    CubeMap irradianceCubeMap = CubeMap(32, 32, CUBE_MAP_HDR);

    // create a prefilter cubemap, and rescale capture FBO to prefilter scale
    CubeMap prefilterCubeMap = CubeMap(256, 256, CUBE_MAP_PREFILTER);

    // generate a 2D LUT from the BRDF equations used
    Texture brdfLUT;
    FullScreenQuad brdfFsQuad;

    Scene() = default;

    void addChildNode(Node* node) {
        children.push_back(node);
    }

    void setEnvMap(CubeMap* envCubeMap) {
        this->envCubeMap = envCubeMap;
    }

    void setAmbientLight(AmbientLight* ambientLight) {
        this->ambientLight = ambientLight;
    }

    void setDirectionalLight(DirectionalLight* directionalLight) {
        this->directionalLight = directionalLight;
    }

    void addPointLight(PointLight* pointLight) {
        pointLights.push_back(pointLight);
    }

    void bindPBREnvMap(Shader &shader) {
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

    void equirectToCubeMap(CubeMap &envCubeMap, Texture &hdrTexture, Shader &equirectToCubeMapShader) {
        captureFramebuffer.createColorAndDepthBuffers(envCubeMap.width, envCubeMap.height);
        captureRenderBuffer.create(envCubeMap.width, envCubeMap.height, GL_DEPTH_COMPONENT24);

        captureFramebuffer.bind();
        envCubeMap.loadFromEquirectTexture(equirectToCubeMapShader, hdrTexture);
        captureFramebuffer.unbind();
    }

    void setupIBL(CubeMap &envCubeMap, Shader &convolutionShader, Shader &prefilterShader, Shader &brdfShader) {
        hasPBREnvMap = true;

        glDisable(GL_BLEND);

        captureFramebuffer.bind();
        captureRenderBuffer.bind();
        captureRenderBuffer.resize(irradianceCubeMap.width, irradianceCubeMap.height);

        captureFramebuffer.bind();
        irradianceCubeMap.convolve(convolutionShader, envCubeMap);
        captureFramebuffer.unbind();

        captureFramebuffer.bind();
        prefilterCubeMap.prefilter(prefilterShader, envCubeMap, captureRenderBuffer);
        captureFramebuffer.unbind();

        TextureCreateParams brdfParams{
            .width = envCubeMap.width,
            .height = envCubeMap.height,
            .internalFormat = GL_RG16F,
            .format = GL_RG,
            .type = GL_FLOAT,
            .wrapS = GL_CLAMP_TO_EDGE,
            .wrapT = GL_CLAMP_TO_EDGE,
            .minFilter = GL_LINEAR,
            .magFilter = GL_LINEAR
        };
        brdfLUT = Texture(brdfParams);
        brdfFsQuad.init();

        captureFramebuffer.bind();
        captureRenderBuffer.bind();
        captureRenderBuffer.resize(prefilterCubeMap.width, prefilterCubeMap.height);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, brdfLUT.ID, 0);

        brdfShader.bind();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        brdfFsQuad.draw();
        brdfShader.unbind();

        captureFramebuffer.unbind();

        glEnable(GL_BLEND);
    }

    static const unsigned int numTextures = 3;
};

#endif // SCENE_H
