#ifndef MATERIAL_H
#define MATERIAL_H

#include <memory>
#include <vector>

#include <Shader.h>
#include <Texture.h>

class Material {
public:
    std::vector<TextureID> textures;

    std::shared_ptr<Shader> shader;

    virtual void bind() = 0;
    virtual void unbind() = 0;

    virtual void cleanup() = 0;
};

#endif // MATERIAL_H
