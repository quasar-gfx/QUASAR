#ifndef MATERIAL_H
#define MATERIAL_H

#include <memory>
#include <vector>

#include <Shaders/Shader.h>
#include <Texture.h>

enum class AlphaMode : uint8_t {
    OPAQUE = 0,
    MASKED,
    TRANSPARENT
};

class Material {
public:
    std::vector<TextureID> textures;
    std::shared_ptr<Shader> shader;

    explicit Material() = default;
    ~Material() {
        for (auto& textureID : textures) {
            if (textureID == 0) continue;
            glDeleteTextures(1, &textureID);
        }
    }

    virtual void bind() const {
        shader->bind();
    }

    virtual unsigned int getTextureCount() const = 0;

    void unbind() const {
        for (int i = 0; i < textures.size(); i++) {
            glActiveTexture(GL_TEXTURE0 + i);
            glBindTexture(GL_TEXTURE_2D, 0);
        }
        shader->unbind();
    }
};

#endif // MATERIAL_H
