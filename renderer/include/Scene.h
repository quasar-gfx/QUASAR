#ifndef SCENE_H
#define SCENE_H

#include <vector>

#include <Buffer.h>
#include <CubeMap.h>
#include <Lights/Lights.h>
#include <RenderTargets/RenderTarget.h>
#include <Primitives/FullScreenQuad.h>
#include <Shaders/Shader.h>

namespace quasar {

class Scene {
public:
    struct GPUPointLightBlock {
        PointLight::GPUPointLight lights[PointLight::maxPointLights];
        int numPointLights;
    };

    glm::vec4 backgroundColor = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    CubeMap* envCubeMap = nullptr;

    AmbientLight* ambientLight = nullptr;
    DirectionalLight* directionalLight = nullptr;
    std::vector<PointLight*> pointLights;

    Node rootNode;

    Buffer pointLightUBO;

    Scene();
    ~Scene() = default;

    void setEnvMap(CubeMap* envCubeMap);
    void setAmbientLight(AmbientLight* ambientLight);
    void setDirectionalLight(DirectionalLight* directionalLight);
    void addPointLight(PointLight* pointLight);
    void addChildNode(Node* node);

    Node* findNodeByName(const std::string& name);

    void updateAnimations(float dt);

    void equirectToCubeMap(const CubeMap& envCubeMap, const Texture& hdrTexture);
    void setupIBL(const CubeMap& envCubeMap);

    int bindMaterial(const Material* material);

    void clear();

    static const uint numTextures = 3;

private:
    bool hasPBREnvMap = false;

    // Create an irradiance cubemap, and rescale capture FBO to irradiance scale
    CubeMap irradianceCubeMap;

    // Create a prefilter cubemap, and rescale capture FBO to prefilter scale
    CubeMap prefilterCubeMap;

    // Generate a 2D LUT from the BRDF equations used
    Texture brdfLUT;
    FullScreenQuad brdfFsQuad;

    // Converts HDR equirectangular environment map to cubemap equivalent
    Shader equirectToCubeMapShader;

    // Solves diffuse integral by convolution to create an irradiance cubemap
    Shader convolutionShader;

    // Runs a quasi monte-carlo simulation on the environment lighting to create a prefilter cubemap
    Shader prefilterShader;

    // BRDF shader
    Shader brdfShader;

    RenderTarget captureRenderTarget;
    Renderbuffer captureRenderBuffer;

    int bindAmbientLight(const Material* material, int texIdx);
    int bindDirectionalLight(const Material* material, int texIdx);
    int bindPointLights(const Material* material, int texIdx);
};

} // namespace quasar

#endif // SCENE_H
