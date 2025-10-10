#ifndef MATERIAL_H
#define MATERIAL_H

#include <memory>
#include <vector>

#include <Shaders/Shader.h>
#include <Texture.h>

namespace quasar {

class Material {
public:
    enum class AlphaMode : uint8_t {
        OPAQUE = 0,
        MASKED,
        TRANSPARENT
    };

    enum class RenderPipelineMode : uint8_t {
        Forward = 0,
        Deferred,
    };

    Material::AlphaMode alphaMode = Material::AlphaMode::OPAQUE;

    Material();
    Material(Material::AlphaMode alphaMode);
    Material(const std::string& name, Material::AlphaMode alphaMode);
    ~Material() = default;

    const std::string& getName() const { return name; }

    void setName(const std::string& name) { this->name = name; }

    virtual void bind() const = 0;

    void unbind() const {
        for (int i = 0; i < textures.size(); i++) {
            glActiveTexture(GL_TEXTURE0 + i);
            glBindTexture(GL_TEXTURE_2D, 0);
        }
    }

    virtual bool isTransparent() const { return alphaMode == Material::AlphaMode::TRANSPARENT; }

    virtual std::shared_ptr<Shader> getShader() const = 0;
    virtual uint getTextureCount() const = 0;

    static void getPipelineMode(RenderPipelineMode& mode) { mode = pipelineMode; }
    static void setPipelineMode(RenderPipelineMode mode) { pipelineMode = mode; }

    static RenderPipelineMode pipelineMode;

    static uint32_t getNextID() { return nextID; }

protected:
    uint32_t ID;
    static uint32_t nextID;

    std::string name;

    std::vector<const Texture*> textures;
};

} // namespace quasar

#endif // MATERIAL_H
