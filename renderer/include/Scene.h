#ifndef SCENE_H
#define SCENE_H

#include <vector>
#include <memory>

#include <Shaders/Shader.h>
#include <Primatives/Entity.h>
#include <Lights/Lights.h>
#include <CubeMap.h>
#include <Primatives/Mesh.h>
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

    explicit Scene() {
        ShaderCreateParams equirectToCubeMapShaderParams = {
            .vertexData = SHADER_CUBEMAP_VERT,
            .vertexDataSize = SHADER_CUBEMAP_VERT_len,
            .fragmentData = SHADER_EQUIRECTANGULAR2CUBEMAP_FRAG,
            .fragmentDataSize = SHADER_EQUIRECTANGULAR2CUBEMAP_FRAG_len
        };
        equirectToCubeMapShader = std::make_shared<Shader>(equirectToCubeMapShaderParams);

        ShaderCreateParams convolutionShaderParams = {
            .vertexData = SHADER_CUBEMAP_VERT,
            .vertexDataSize = SHADER_CUBEMAP_VERT_len,
            .fragmentData = SHADER_IRRADIANCECONVOLUTION_FRAG,
            .fragmentDataSize = SHADER_IRRADIANCECONVOLUTION_FRAG_len
        };
        convolutionShader = std::make_shared<Shader>(convolutionShaderParams);

        ShaderCreateParams prefilterShaderParams = {
            .vertexData = SHADER_CUBEMAP_VERT,
            .vertexDataSize = SHADER_CUBEMAP_VERT_len,
            .fragmentData = SHADER_PREFILTER_FRAG,
            .fragmentDataSize = SHADER_PREFILTER_FRAG_len
        };
        prefilterShader = std::make_shared<Shader>(prefilterShaderParams);

        ShaderCreateParams brdfShaderParams = {
            .vertexData = SHADER_BRDF_VERT,
            .vertexDataSize = SHADER_BRDF_VERT_len,
            .fragmentData = SHADER_BRDF_FRAG,
            .fragmentDataSize = SHADER_BRDF_FRAG_len
        };
        brdfShader = std::make_shared<Shader>(brdfShaderParams);
    }

    void addChildNode(Node* node);

    void setEnvMap(CubeMap* envCubeMap);

    void setAmbientLight(AmbientLight* ambientLight);
    void setDirectionalLight(DirectionalLight* directionalLight);
    void addPointLight(PointLight* pointLight);

    void bindPBREnvMap(Shader &shader);
    void equirectToCubeMap(CubeMap &envCubeMap, Texture &hdrTexture);
    void setupIBL(CubeMap &envCubeMap);

    static const unsigned int numTextures = 3;
};

#endif // SCENE_H
