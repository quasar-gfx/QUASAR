#ifndef MATERIAL_H
#define MATERIAL_H

#include <vector>

#include <CubeMap.h>
#include <Shader.h>
#include <Texture.h>
#include <Lights.h>

struct MaterialCreateParams {
    std::string albedoTexturePath = "";
    std::string specularTexturePath = "";
    std::string normalTexturePath = "";
    std::string metallicTexturePath = "";
    std::string roughnessTexturePath = "";
    std::string aoTexturePath = "";
    TextureID albedoTextureID;
    TextureID specularTextureID;
    TextureID normalTextureID;
    TextureID metallicTextureID;
    TextureID roughnessTextureID;
    TextureID aoTextureID;
    float shininess = 1.0;
};

class Material {
public:
    float empty = true;

    std::vector<TextureID> textures;
    float shininess;

    Material() = default;

    Material(const MaterialCreateParams &params) {
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

        if (params.specularTexturePath != "") {
            textureParams.path = params.specularTexturePath;
            Texture texture = Texture(textureParams);
            textures.push_back(texture.ID);
        }
        else {
            textures.push_back(params.specularTextureID);
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

        shininess = params.shininess;
        empty = false;
    }

    void bind(Shader &shader);
    void unbind();

    void cleanup();

    static const unsigned int numTextures = 6;
};

#endif // MATERIAL_H
