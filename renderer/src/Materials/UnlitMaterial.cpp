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

    ShaderDataCreateParams UnlitMaterialParams{
        .vertexCodeData = SHADER_COMMON_VERT,
        .vertexCodeSize = SHADER_COMMON_VERT_len,
        .fragmentCodeData = SHADER_MATERIAL_UNLIT_FRAG,
        .fragmentCodeSize = SHADER_MATERIAL_UNLIT_FRAG_len
    };
    shader = std::make_shared<Shader>(UnlitMaterialParams);

    color = params.color;
    opacity = params.opacity;
    transparent = params.transparent;
}

void UnlitMaterial::bind() const {
    shader->bind();
    shader->setVec3("material.baseColor", color);
    shader->setFloat("material.opacity", opacity);
    shader->setBool("material.transparent", transparent);

    std::string name = "material.diffuseMap";
    glActiveTexture(GL_TEXTURE0);

    shader->setInt(name, 0);
    glBindTexture(GL_TEXTURE_2D, textures[0]);
}
