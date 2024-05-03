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

    ShaderCreateParams pbrShaderParams{
        .vertexCodeData = SHADER_MAIN_VERT,
        .vertexCodeSize = SHADER_MAIN_VERT_len,
        .fragmentCodeData = SHADER_PBR_FRAG,
        .fragmentCodeSize = SHADER_PBR_FRAG_len
    };
    shader = std::make_unique<Shader>(pbrShaderParams);

    metalRoughnessCombined = params.metalRoughnessCombined;
    transparent = params.transparent;
}

void PBRMaterial::bind() {
    shader->bind();
    shader->setFloat("shininess", shininess);
    shader->setBool("metalRoughnessCombined", metalRoughnessCombined);
    shader->setBool("transparent", transparent);

    std::string name;
    for (int i = 0; i < textures.size(); i++) {
        glActiveTexture(GL_TEXTURE0 + i);
        switch(i) {
        case 0:
            name = "albedoMap";
            break;
        case 1:
            name = "normalMap";
            shader->setBool("normalMapped", textures[i] != 0);
            break;
        case 2:
            name = "metallicMap";
            break;
        case 3:
            name = "roughnessMap";
            break;
        case 4:
            name = "aoMap";
            shader->setBool("aoMapped", textures[i] != 0);
            break;
        default:
            break;
        }

        shader->setInt(name, i);
        glBindTexture(GL_TEXTURE_2D, textures[i]);
    }
}
