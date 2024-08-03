#include <Materials/UnlitMaterial.h>

UnlitMaterial::UnlitMaterial(const UnlitMaterialCreateParams &params)
        : baseColor(params.baseColor)
        , baseColorFactor(params.baseColorFactor)
        , alphaMode(params.alphaMode)
        , maskThreshold(params.maskThreshold) {
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

    ShaderDataCreateParams unlitShaderParams{
        .vertexCodeData = SHADER_COMMON_VERT,
        .vertexCodeSize = SHADER_COMMON_VERT_len,
        .fragmentCodeData = SHADER_MATERIAL_UNLIT_FRAG,
        .fragmentCodeSize = SHADER_MATERIAL_UNLIT_FRAG_len,
        .defines = {
            "#define ALPHA_OPAQUE " + std::to_string(static_cast<uint8_t>(AlphaMode::OPAQUE)),
            "#define ALPHA_MASK " + std::to_string(static_cast<uint8_t>(AlphaMode::MASKED)),
            "#define ALPHA_BLEND " + std::to_string(static_cast<uint8_t>(AlphaMode::TRANSPARENT))
        }
    };
    shader = std::make_shared<Shader>(unlitShaderParams);
}

void UnlitMaterial::bind() const {
    shader->bind();
    shader->setVec4("material.baseColor", baseColor);
    shader->setVec4("material.baseColorFactor", baseColorFactor);
    shader->setInt("material.alphaMode", static_cast<int>(alphaMode));
    shader->setFloat("material.maskThreshold", maskThreshold);

    std::string name = "material.baseColorMap";
    glActiveTexture(GL_TEXTURE0);
    shader->setBool("material.hasBaseColorMap", textures[0] != 0);

    shader->setInt(name, 0);
    glBindTexture(GL_TEXTURE_2D, textures[0]);
}
