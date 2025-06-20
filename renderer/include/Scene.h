#ifndef SCENE_H
#define SCENE_H

#include <vector>

#include <Shaders/Shader.h>
#include <Lights/Lights.h>
#include <CubeMap.h>
#include <Primitives/Mesh.h>
#include <RenderTargets/RenderTarget.h>
#include <Primitives/FullScreenQuad.h>

namespace quasar {

class Scene {
public:
    CubeMap* envCubeMap = nullptr;
    AmbientLight* ambientLight = nullptr;
    DirectionalLight* directionalLight = nullptr;
    std::vector<PointLight*> pointLights;

    Node rootNode;

    bool hasPBREnvMap = false;

    glm::vec4 backgroundColor = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);

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

    Scene();

    void updateAnimations(float dt);

    void addChildNode(Node* node);

    void setEnvMap(CubeMap* envCubeMap);

    void setAmbientLight(AmbientLight* ambientLight);
    void setDirectionalLight(DirectionalLight* directionalLight);
    void addPointLight(PointLight* pointLight);

    void equirectToCubeMap(const CubeMap& envCubeMap, const Texture& hdrTexture);
    void setupIBL(const CubeMap& envCubeMap);

    void bindMaterial(const Material* material) const;

    void clear();

    Node* findNodeByName(const std::string& name);

    static const uint numTextures = 3;

private:
    RenderTarget captureRenderTarget;
    Renderbuffer captureRenderBuffer;
};

} // namespace quasar

#endif // SCENE_H
