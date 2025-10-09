#ifndef MATERIAL_H
#define MATERIAL_H

#include <memory>
#include <vector>

#include <Shaders/Shader.h>
#include <Texture.h>

namespace quasar {

enum class AlphaMode : uint8_t {
    OPAQUE = 0,
    MASKED,
    TRANSPARENT
};

class Material {
public:
    AlphaMode alphaMode = AlphaMode::OPAQUE;

    Material() = default;
    Material(AlphaMode alphaMode)
        : ID(nextID++)
        , alphaMode(alphaMode)
    {}
    ~Material() = default;

    virtual void bind() const = 0;

    void unbind() const {
        for (int i = 0; i < textures.size(); i++) {
            glActiveTexture(GL_TEXTURE0 + i);
            glBindTexture(GL_TEXTURE_2D, 0);
        }
    }

    virtual bool isTransparent() const { return alphaMode == AlphaMode::TRANSPARENT; }

    virtual std::shared_ptr<Shader> getShader() const = 0;
    virtual uint getTextureCount() const = 0;

protected:
    uint32_t ID;
    static uint32_t nextID;

    std::vector<const Texture*> textures;
};

} // namespace quasar

#endif // MATERIAL_H
