#include <Lights/PointLight.h>
#include <Materials/LitMaterial.h>

using namespace quasar;

std::shared_ptr<Shader> LitMaterial::shader = nullptr;
std::shared_ptr<Shader> LitMaterial::deferredShader = nullptr;
std::shared_ptr<Shader> LitMaterial::forwardShader = nullptr;
Material::RenderPipelineMode Material::pipelineMode = Material::RenderPipelineMode::Deferred;
std::vector<std::string> LitMaterial::extraShaderDefines;

void LitMaterial::setPipelineMode(RenderPipelineMode mode) {
    Material::setPipelineMode(mode);
    shader = (mode == RenderPipelineMode::Forward && forwardShader != nullptr) ? forwardShader : deferredShader;
}

LitMaterial::LitMaterial(const LitMaterialCreateParams& params)
    : baseColor(params.baseColor)
    , baseColorFactor(params.baseColorFactor)
    , maskThreshold(params.maskThreshold)
    , emissiveFactor(params.emissiveFactor)
    , metallic(params.metallic)
    , metallicFactor(params.metallicFactor)
    , roughness(params.roughness)
    , roughnessFactor(params.roughnessFactor)
    , metalRoughnessCombined(params.metalRoughnessCombined)
    , Material(params.name, params.alphaMode)
{
    TextureFileCreateParams textureParams{
        .wrapS = GL_REPEAT,
        .wrapT = GL_REPEAT,
        .minFilter = GL_LINEAR_MIPMAP_LINEAR,
        .magFilter = GL_LINEAR,
    };

    if (params.albedoTexturePath != "") {
        textureParams.path = params.albedoTexturePath;
        Texture* texture = new Texture(textureParams);
        textures.push_back(texture);
    }
    else {
        textures.push_back(params.albedoTexture);
    }

    // Only gamma correct color textures
    textureParams.gammaCorrected = false;

    if (params.normalTexturePath != "") {
        textureParams.path = params.normalTexturePath;
        Texture* texture = new Texture(textureParams);
        textures.push_back(texture);
    }
    else {
        textures.push_back(params.normalTexture);
    }

    if (params.metallicTexturePath != "") {
        textureParams.path = params.metallicTexturePath;
        Texture* texture = new Texture(textureParams);
        textures.push_back(texture);
    }
    else {
        textures.push_back(params.metallicTexture);
    }

    if (params.roughnessTexturePath != "") {
        textureParams.path = params.roughnessTexturePath;
        Texture* texture = new Texture(textureParams);
        textures.push_back(texture);
    }
    else {
        textures.push_back(params.roughnessTexture);
    }

    if (params.aoTexturePath != "") {
        textureParams.path = params.aoTexturePath;
        Texture* texture = new Texture(textureParams);
        textures.push_back(texture);
    }
    else {
        textures.push_back(params.aoTexture);
    }

    if (params.emissiveTexturePath != "") {
        textureParams.path = params.emissiveTexturePath;
        Texture* texture = new Texture(textureParams);
        textures.push_back(texture);
    }
    else {
        textures.push_back(params.emissiveTexture);
    }

    // Build shader programs lazily; maintain both forward and deferred variants when available
    if (deferredShader == nullptr || forwardShader == nullptr) {
        std::vector<std::string> defines = {
            "#define MAX_POINT_LIGHTS " + std::to_string(PointLight::maxPointLights),
            "#define ALPHA_OPAQUE " + std::to_string(static_cast<uint8_t>(Material::AlphaMode::OPAQUE)),
            "#define ALPHA_MASK " + std::to_string(static_cast<uint8_t>(Material::AlphaMode::MASKED)),
            "#define ALPHA_BLEND " + std::to_string(static_cast<uint8_t>(Material::AlphaMode::TRANSPARENT))
        };
        for (const auto& define : extraShaderDefines) {
            defines.push_back(define);
        }

        // Forward shader (always available for both GLES and GL)
        ShaderDataCreateParams forwardParams{
            .vertexCodeData = SHADER_BUILTIN_COMMON_VERT,
            .vertexCodeSize = SHADER_BUILTIN_COMMON_VERT_len,
            .fragmentCodeData = SHADER_BUILTIN_MATERIAL_LIT_FRAG,
            .fragmentCodeSize = SHADER_BUILTIN_MATERIAL_LIT_FRAG_len,
#ifndef GL_CORE
            .extensions = {
                "#extension GL_EXT_texture_cube_map_array : enable"
            },
#endif
            .defines = defines
        };
        forwardShader = std::make_shared<Shader>(forwardParams);

#ifdef GL_CORE
        ShaderDataCreateParams deferredParams{
            .vertexCodeData = SHADER_BUILTIN_COMMON_VERT,
            .vertexCodeSize = SHADER_BUILTIN_COMMON_VERT_len,
            .fragmentCodeData = SHADER_BUILTIN_DEFERRED_GBUFFER_FRAG,
            .fragmentCodeSize = SHADER_BUILTIN_DEFERRED_GBUFFER_FRAG_len,
            .defines = defines
        };
        deferredShader = std::make_shared<Shader>(deferredParams);
#else
        deferredShader = forwardShader;
#endif

        // Initialize alias according to current pipeline mode
        shader = (pipelineMode == RenderPipelineMode::Forward) ? forwardShader : deferredShader;
    }
}

bool LitMaterial::isTransparent() const {
    return alphaMode == Material::AlphaMode::TRANSPARENT || baseColor.a * baseColorFactor.a < 1.0f;
}

void LitMaterial::bind() const {
    shader->bind();
    shader->setVec4("material.baseColor", baseColor);
    shader->setVec4("material.baseColorFactor", baseColorFactor);
    shader->setInt("material.alphaMode", static_cast<int>(alphaMode));
    shader->setFloat("material.maskThreshold", maskThreshold);
    shader->setVec3("material.emissiveFactor", emissiveFactor);
    shader->setFloat("material.metallic", metallic);
    shader->setFloat("material.metallicFactor", metallicFactor);
    shader->setFloat("material.roughness", roughness);
    shader->setFloat("material.roughnessFactor", roughnessFactor);
    shader->setBool("material.metalRoughnessCombined", metalRoughnessCombined);

    std::string name;
    for (int i = 0; i < textures.size(); i++) {
        glActiveTexture(GL_TEXTURE0 + i);
        switch (i) {
        case 0:
            name = "material.baseColorMap";
            shader->setBool("material.hasBaseColorMap", textures[i] != nullptr);
            break;
        case 1:
            name = "material.normalMap";
            shader->setBool("material.hasNormalMap", textures[i] != nullptr);
            break;
        case 2:
            name = "material.metallicMap";
            shader->setBool("material.hasMetallicMap", textures[i] != nullptr);
            break;
        case 3:
            name = "material.roughnessMap";
            shader->setBool("material.hasRoughnessMap", !metalRoughnessCombined ? textures[i] != nullptr : textures[i-1] != nullptr);
            break;
        case 4:
            name = "material.aoMap";
            shader->setBool("material.hasAOMap", textures[i] != nullptr);
            break;
        case 5:
            name = "material.emissiveMap";
            shader->setBool("material.hasEmissiveMap", textures[i] != nullptr);
            break;
        default:
            break;
        }

        if (textures[i] != nullptr) {
            shader->setTexture(name, *textures[i], i);
        }
        else {
            shader->setInt(name, i);
            glBindTexture(GL_TEXTURE_2D, 0);
        }
    }
}
