#ifndef SCENE_H
#define SCENE_H

#include <vector>
#include <memory>

#include <Shaders/Shader.h>
#include <Primatives/Entity.h>
#include <Lights/Lights.h>
#include <CubeMap.h>
#include <Primatives/Mesh.h>
#include <RenderTargets/RenderTarget.h>
#include <FullScreenQuad.h>

class Scene {
public:
    CubeMap* envCubeMap = nullptr;
    AmbientLight* ambientLight = nullptr;
    DirectionalLight* directionalLight = nullptr;
    std::vector<PointLight*> pointLights;

    std::vector<Node*> children;

    bool hasPBREnvMap = false;

    glm::vec4 backgroundColor = glm::vec4(0.1f, 0.1f, 0.1f, 1.0f);

    // create an irradiance cubemap, and rescale capture FBO to irradiance scale
    CubeMap irradianceCubeMap = CubeMap({ .width = 32, .height = 32, .type = CubeMapType::STANDARD });

    // create a prefilter cubemap, and rescale capture FBO to prefilter scale
    CubeMap prefilterCubeMap = CubeMap({ .width = 256, .height = 256, .type = CubeMapType::PREFILTER });

    // generate a 2D LUT from the BRDF equations used
    Texture brdfLUT;
    FullScreenQuad brdfFsQuad;

    // converts HDR equirectangular environment map to cubemap equivalent
    std::shared_ptr<Shader> equirectToCubeMapShader;

    // solves diffuse integral by convolution to create an irradiance cubemap
    std::shared_ptr<Shader> convolutionShader;

    // runs a quasi monte-carlo simulation on the environment lighting to create a prefilter cubemap
    std::shared_ptr<Shader> prefilterShader;

    // BRDF shader
    std::shared_ptr<Shader> brdfShader;

    explicit Scene();

    void addChildNode(Node* node);

    void setEnvMap(CubeMap* envCubeMap);

    void setAmbientLight(AmbientLight* ambientLight);
    void setDirectionalLight(DirectionalLight* directionalLight);
    void addPointLight(PointLight* pointLight);

    void equirectToCubeMap(CubeMap &envCubeMap, Texture &hdrTexture);
    void setupIBL(CubeMap &envCubeMap);

    void bindMaterial(Material* material);

    static const unsigned int numTextures = 3;

private:
    RenderTarget captureRenderTarget;
    Renderbuffer captureRenderBuffer;
};

#endif // SCENE_H
