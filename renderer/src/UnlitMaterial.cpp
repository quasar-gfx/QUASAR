#include <Materials/UnlitMaterial.h>

UnlitMaterial::UnlitMaterial(const UnlitMaterialCreateParams &params) {
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

    ShaderCreateParams UnlitMaterialParams{
        .vertexCodeData = SHADER_COMMON_VERT,
        .vertexCodeSize = SHADER_COMMON_VERT_len,
        .fragmentCodeData = SHADER_UNLIT_FRAG,
        .fragmentCodeSize = SHADER_UNLIT_FRAG_len
    };
    shader = std::make_shared<Shader>(UnlitMaterialParams);

    transparent = params.transparent;
}

void UnlitMaterial::bind() {
    shader->bind();
    shader->setBool("transparent", transparent);

    std::string name = "diffuseMap";
    glActiveTexture(GL_TEXTURE0);

    shader->setInt(name, 0);
    glBindTexture(GL_TEXTURE_2D, textures[0]);
}
