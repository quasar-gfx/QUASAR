#ifndef MATERIAL_H
#define MATERIAL_H

#include <memory>
#include <vector>

#include <Shaders/Shader.h>
#include <Texture.h>

class Material {
public:
    std::vector<TextureID> textures;

    std::shared_ptr<Shader> shader;

    virtual void bind() = 0;

    void unbind() {
        for (int i = 0; i < textures.size(); i++) {
            glActiveTexture(GL_TEXTURE0 + i);
            glBindTexture(GL_TEXTURE_2D, 0);
        }
        shader->unbind();
    }

    void cleanup() {
        for (auto &textureID : textures) {
            if (textureID == 0) continue;
            glDeleteTextures(1, &textureID);
        }
    }
};

#endif // MATERIAL_H
