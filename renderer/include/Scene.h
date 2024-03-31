#ifndef SCENE_H
#define SCENE_H

#include <vector>

#include <Entity.h>
#include <Lights.h>
#include <CubeMap.h>
#include <Mesh.h>
#include <Framebuffer.h>
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
    Framebuffer captureFramebuffer;
    Renderbuffer captureRenderBuffer;

    // create an irradiance cubemap, and rescale capture FBO to irradiance scale
    CubeMap irradianceCubeMap = CubeMap(32, 32, CUBE_MAP_HDR);

    // create a prefilter cubemap, and rescale capture FBO to prefilter scale
    CubeMap prefilterCubeMap = CubeMap(256, 256, CUBE_MAP_PREFILTER);

    // generate a 2D LUT from the BRDF equations used
    Texture brdfLUT;
    FullScreenQuad brdfFsQuad;

    Scene() = default;

    void addChildNode(Node* node);

    void setEnvMap(CubeMap* envCubeMap);

    void setAmbientLight(AmbientLight* ambientLight);
    void setDirectionalLight(DirectionalLight* directionalLight);
    void addPointLight(PointLight* pointLight);

    void bindPBREnvMap(Shader &shader);
    void equirectToCubeMap(CubeMap &envCubeMap, Texture &hdrTexture, Shader &equirectToCubeMapShader);
    void setupIBL(CubeMap &envCubeMap, Shader &convolutionShader, Shader &prefilterShader, Shader &brdfShader);

    static const unsigned int numTextures = 3;
};

#endif // SCENE_H
