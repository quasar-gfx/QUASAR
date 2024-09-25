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
    std::vector<Texture*> textures;

    Material() = default;
    ~Material() = default;

    virtual void bind() const = 0;

    void unbind() const {
        for (int i = 0; i < textures.size(); i++) {
            glActiveTexture(GL_TEXTURE0 + i);
            glBindTexture(GL_TEXTURE_2D, 0);
        }
    }

    virtual Shader* getShader() const = 0;

    virtual unsigned int getTextureCount() const = 0;
};

#endif // MATERIAL_H
