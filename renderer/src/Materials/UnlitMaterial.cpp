#include <Materials/UnlitMaterial.h>

Shader* UnlitMaterial::shader = nullptr;

UnlitMaterial::UnlitMaterial(const UnlitMaterialCreateParams &params)
        : baseColor(params.baseColor)
        , baseColorFactor(params.baseColorFactor)
        , alphaMode(params.alphaMode)
        , maskThreshold(params.maskThreshold) {
    TextureFileCreateParams textureParams{
        .wrapS = GL_REPEAT,
        .wrapT = GL_REPEAT,
        .minFilter = GL_LINEAR_MIPMAP_LINEAR,
        .magFilter = GL_LINEAR
    };

    if (params.baseColorTexturePath != "") {
        textureParams.path = params.baseColorTexturePath;
        Texture* texture = new Texture(textureParams);
        textures.push_back(texture);
    }
    else {
        textures.push_back(params.baseColorTexture);
    }

    if (shader == nullptr) {
        ShaderDataCreateParams unlitShaderParams{
            .vertexCodeData = SHADER_BUILTIN_COMMON_VERT,
            .vertexCodeSize = SHADER_BUILTIN_COMMON_VERT_len,
            .fragmentCodeData = SHADER_BUILTIN_MATERIAL_UNLIT_FRAG,
            .fragmentCodeSize = SHADER_BUILTIN_MATERIAL_UNLIT_FRAG_len,
            .defines = {
                "#define ALPHA_OPAQUE " + std::to_string(static_cast<uint8_t>(AlphaMode::OPAQUE)),
                "#define ALPHA_MASK " + std::to_string(static_cast<uint8_t>(AlphaMode::MASKED)),
                "#define ALPHA_BLEND " + std::to_string(static_cast<uint8_t>(AlphaMode::TRANSPARENT))
            }
        };
        shader = new Shader(unlitShaderParams);
    }
}

void UnlitMaterial::bind() const {
    shader->bind();
    shader->setVec4("material.baseColor", baseColor);
    shader->setVec4("material.baseColorFactor", baseColorFactor);
    shader->setInt("material.alphaMode", static_cast<int>(alphaMode));
    shader->setFloat("material.maskThreshold", maskThreshold);

    std::string name = "material.baseColorMap";
    glActiveTexture(GL_TEXTURE0);
    shader->setBool("material.hasBaseColorMap", textures[0] != nullptr);

    if (textures[0] != nullptr) {
        shader->setTexture(name, *textures[0], 0);
    }
    else {
        shader->setInt(name, 0);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
}

UnlitMaterial::~UnlitMaterial() {
    if (shader != nullptr) {
        delete shader;
        shader = nullptr;
    }
}
