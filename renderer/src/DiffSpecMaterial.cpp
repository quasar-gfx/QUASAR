#include <Materials/DiffSpecMaterial.h>

DiffSpecMaterial::DiffSpecMaterial(const DiffSpecMaterialCreateParams &params) {
    TextureCreateParams textureParams{
        .wrapS = GL_REPEAT,
        .wrapT = GL_REPEAT,
        .minFilter = GL_LINEAR_MIPMAP_LINEAR,
        .magFilter = GL_LINEAR
    };

    if (params.diffuseTexturePath != "") {
        textureParams.path = params.diffuseTexturePath;
        Texture texture = Texture(textureParams);
        textures.push_back(texture.ID);
    }
    else {
        textures.push_back(params.diffuseTextureID);
    }

    if (params.specularTexturePath != "") {
        textureParams.path = params.specularTexturePath;
        Texture texture = Texture(textureParams);
        textures.push_back(texture.ID);
    }
    else {
        textures.push_back(params.specularTextureID);
    }

    shininess = params.shininess;
}

void DiffSpecMaterial::bind(Shader &shader) {
    shader.setFloat("shininess", shininess);

    std::string name;
    for (int i = 0; i < textures.size(); i++) {
        glActiveTexture(GL_TEXTURE0 + i);
        switch(i) {
        case 0:
            name = "diffuseMap";
            break;
        case 1:
            name = "specularMap";
            break;
        default:
            break;
        }

        shader.setInt(name, i);
        glBindTexture(GL_TEXTURE_2D, textures[i]);
    }
}

void DiffSpecMaterial::unbind() {
    for (int i = 0; i < textures.size(); i++) {
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
}

void DiffSpecMaterial::cleanup() {
    for (auto &textureID : textures) {
        if (textureID == 0) continue;
        glDeleteTextures(1, &textureID);
    }
}
