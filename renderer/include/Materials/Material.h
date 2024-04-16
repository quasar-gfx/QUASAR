#ifndef MATERIAL_H
#define MATERIAL_H

#include <vector>

#include <Shader.h>
#include <Texture.h>

class Material {
public:
    std::vector<TextureID> textures;

    virtual void bind(Shader &shader) = 0;
    virtual void unbind() = 0;

    virtual void cleanup() = 0;
};

#endif // MATERIAL_H
