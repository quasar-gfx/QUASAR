#include <Lights/PointLight.h>
#include <Materials/LitMaterial.h>

using namespace quasar;

Shader* LitMaterial::shader = nullptr;
std::vector<std::string> LitMaterial::extraShaderDefines;

LitMaterial::LitMaterial(const LitMaterialCreateParams& params)
    : baseColor(params.baseColor)
    , baseColorFactor(params.baseColorFactor)
    , alphaMode(params.alphaMode)
    , maskThreshold(params.maskThreshold)
    , emissiveFactor(params.emissiveFactor)
    , metallic(params.metallic)
    , metallicFactor(params.metallicFactor)
    , roughness(params.roughness)
    , roughnessFactor(params.roughnessFactor)
    , metalRoughnessCombined(params.metalRoughnessCombined)
{
    TextureFileCreateParams textureParams{
        .wrapS = GL_REPEAT,
        .wrapT = GL_REPEAT,
        .minFilter = GL_LINEAR_MIPMAP_LINEAR,
        .magFilter = GL_LINEAR
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

    if (shader == nullptr) {
        std::vector<std::string> defines = {
            "#define MAX_POINT_LIGHTS " + std::to_string(PointLight::maxPointLights),
            "#define ALPHA_OPAQUE " + std::to_string(static_cast<uint8_t>(AlphaMode::OPAQUE)),
            "#define ALPHA_MASK " + std::to_string(static_cast<uint8_t>(AlphaMode::MASKED)),
            "#define ALPHA_BLEND " + std::to_string(static_cast<uint8_t>(AlphaMode::TRANSPARENT))
        };
        for (const auto& define : extraShaderDefines) {
            defines.push_back(define);
        }

        ShaderDataCreateParams pbrShaderParams{
            .vertexCodeData = SHADER_BUILTIN_COMMON_VERT,
            .vertexCodeSize = SHADER_BUILTIN_COMMON_VERT_len,
            .fragmentCodeData = SHADER_BUILTIN_DEFERRED_GBUFFER_FRAG,
            .fragmentCodeSize = SHADER_BUILTIN_DEFERRED_GBUFFER_FRAG_len,
#ifdef GL_ES
            .extensions = {
                "#extension GL_EXT_texture_cube_map_array : enable"
            },
#endif
            .defines = defines
        };
        shader = new Shader(pbrShaderParams);
    }
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

LitMaterial::~LitMaterial() {
    if (shader != nullptr) {
        delete shader;
        shader = nullptr;
    }
}
