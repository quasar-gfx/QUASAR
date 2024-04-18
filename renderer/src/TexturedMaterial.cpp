#include <Materials/TexturedMaterial.h>

TexturedMaterial::TexturedMaterial(const TexturedMaterialCreateParams &params) {
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

    ShaderCreateParams texturedMaterialParams{
        .vertexData = SHADER_DIFFUSESPECULAR_VERT,
        .vertexDataSize = SHADER_DIFFUSESPECULAR_VERT_len,
        .fragmentData = SHADER_TEXTURED_FRAG,
        .fragmentDataSize = SHADER_TEXTURED_FRAG_len
    };
    shader = std::make_shared<Shader>(texturedMaterialParams);
}

void TexturedMaterial::bind() {
    shader->bind();

    std::string name = "diffuseMap";
    glActiveTexture(GL_TEXTURE0);

    shader->setInt(name, 0);
    glBindTexture(GL_TEXTURE_2D, textures[0]);
}
