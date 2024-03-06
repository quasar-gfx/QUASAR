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
    FrameBuffer captureFramebuffer = FrameBuffer(512, 512);
    RenderBuffer captureRenderBuffer = RenderBuffer(512, 512, GL_DEPTH_COMPONENT24);

    // create an irradiance cubemap, and rescale capture FBO to irradiance scale
    CubeMap irradianceCubeMap = CubeMap(32, 32, CUBE_MAP_HDR);

    // create a prefilter cubemap, and rescale capture FBO to prefilter scale
    CubeMap prefilterCubeMap = CubeMap(256, 256, CUBE_MAP_PREFILTER);

    // generate a 2D LUT from the BRDF equations used
    Texture brdfLUT = Texture(512, 512, GL_RG16F, GL_RG, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_LINEAR, GL_LINEAR);
    FullScreenQuad brdfFsQuad = FullScreenQuad();

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

    void setupIBL(CubeMap &envCubeMap, Texture &hdrTexture, Shader &equirectToCubeMapShader, Shader &convolutionShader, Shader &prefilterShader, Shader &brdfShader) {
        hasPBREnvMap = true;

        captureFramebuffer.bind();
        envCubeMap.loadFromEquirectTexture(equirectToCubeMapShader, hdrTexture);
        captureFramebuffer.unbind();

        captureFramebuffer.bind();
        captureRenderBuffer.bind();
        captureRenderBuffer.resize(32, 32);

        captureFramebuffer.bind();
        irradianceCubeMap.convolve(convolutionShader, envCubeMap);
        captureFramebuffer.unbind();

        captureFramebuffer.bind();
        prefilterCubeMap.prefilter(prefilterShader, envCubeMap, captureRenderBuffer);
        captureFramebuffer.unbind();

        captureFramebuffer.bind();
        captureRenderBuffer.bind();
        captureRenderBuffer.resize(512, 512);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, brdfLUT.ID, 0);

        glViewport(0, 0, 512, 512);
        brdfShader.bind();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        brdfFsQuad.draw();
        brdfShader.unbind();

        captureFramebuffer.unbind();
    }
};

#endif // SCENE_H
