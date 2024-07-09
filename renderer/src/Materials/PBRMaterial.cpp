#include <Materials/PBRMaterial.h>

PBRMaterial::PBRMaterial(const PBRMaterialCreateParams &params) {
    TextureCreateParams textureParams{
        .wrapS = GL_REPEAT,
        .wrapT = GL_REPEAT,
        .minFilter = GL_LINEAR_MIPMAP_LINEAR,
        .magFilter = GL_LINEAR
    };

    if (params.albedoTexturePath != "") {
        textureParams.path = params.albedoTexturePath;
        Texture texture = Texture(textureParams);
        textures.push_back(texture.ID);
    }
    else {
        textures.push_back(params.albedoTextureID);
    }

    // only gamma correct color textures
    textureParams.gammaCorrected = false;

    if (params.normalTexturePath != "") {
        textureParams.path = params.normalTexturePath;
        Texture texture = Texture(textureParams);
        textures.push_back(texture.ID);
    }
    else {
        textures.push_back(params.normalTextureID);
    }

    if (params.metallicTexturePath != "") {
        textureParams.path = params.metallicTexturePath;
        Texture texture = Texture(textureParams);
        textures.push_back(texture.ID);
    }
    else {
        textures.push_back(params.metallicTextureID);
    }

    if (params.roughnessTexturePath != "") {
        textureParams.path = params.roughnessTexturePath;
        Texture texture = Texture(textureParams);
        textures.push_back(texture.ID);
    }
    else {
        textures.push_back(params.roughnessTextureID);
    }

    if (params.aoTexturePath != "") {
        textureParams.path = params.aoTexturePath;
        Texture texture = Texture(textureParams);
        textures.push_back(texture.ID);
    }
    else {
        textures.push_back(params.aoTextureID);
    }

    ShaderDataCreateParams pbrShaderParams{
        .vertexCodeData = SHADER_COMMON_VERT,
        .vertexCodeSize = SHADER_COMMON_VERT_len,
        .fragmentCodeData = SHADER_MATERIAL_PBR_FRAG,
        .fragmentCodeSize = SHADER_MATERIAL_PBR_FRAG_len,
        .defines = {"#define MAX_POINT_LIGHTS " + std::to_string(params.numPointLights)}
    };
    shader = std::make_unique<Shader>(pbrShaderParams);

    color = params.color;
    opacity = params.opacity;
    transparent = params.transparent;
    maskThreshold = params.maskThreshold;
    metallic = params.metallic;
    roughness = params.roughness;
    metalRoughnessCombined = params.metalRoughnessCombined;
}

void PBRMaterial::bind() const {
    shader->bind();
    shader->setVec3("material.baseColor", color);
    shader->setFloat("material.opacity", opacity);
    shader->setBool("material.transparent", transparent);
    shader->setFloat("material.maskThreshold", maskThreshold);
    shader->setFloat("material.metallic", metallic);
    shader->setFloat("material.roughness", roughness);
    shader->setBool("material.metalRoughnessCombined", metalRoughnessCombined);

    std::string name;
    for (int i = 0; i < textures.size(); i++) {
        glActiveTexture(GL_TEXTURE0 + i);
        switch(i) {
        case 0:
            name = "material.albedoMap";
            break;
        case 1:
            name = "material.normalMap";
            shader->setBool("material.normalMapped", textures[i] != 0);
            break;
        case 2:
            name = "material.metallicMap";
            break;
        case 3:
            name = "material.roughnessMap";
            break;
        case 4:
            name = "material.aoMap";
            shader->setBool("material.aoMapped", textures[i] != 0);
            break;
        default:
            break;
        }

        shader->setInt(name, i);
        glBindTexture(GL_TEXTURE_2D, textures[i]);
    }
}
