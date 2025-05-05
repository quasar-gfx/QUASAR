#ifndef SHADER_BASE_H
#define SHADER_BASE_H

#include <any>
#include <unordered_map>
#include <iostream>
#include <vector>
#include <glm/glm.hpp>
#include <OpenGLObject.h>
#include <Texture.h>

namespace quasar {

enum class ShaderType {
    PROGRAM,
    VERTEX,
    FRAGMENT,
    GEOMETRY,
    COMPUTE
};

class ShaderBase : public OpenGLObject {
public:
    ShaderBase();
    ~ShaderBase() override;

    void bind() const override;
    void unbind() const override;

    void setBool(const std::string &name, bool value) const;
    void setUint(const std::string &name, uint value) const;
    void setInt(const std::string &name, int value) const;
    void setFloat(const std::string &name, float value) const;
    void setVec2(const std::string &name, const glm::vec2 &value) const;
    void setVec3(const std::string &name, const glm::vec3 &value) const;
    void setVec4(const std::string &name, const glm::vec4 &value) const;
    void setMat2(const std::string &name, const glm::mat2 &mat) const;
    void setMat3(const std::string &name, const glm::mat3 &mat) const;
    void setMat4(const std::string &name, const glm::mat4 &mat) const;
    void setTexture(const Texture &texture, int slot) const;
    void setTexture(const std::string &name, const Texture &texture, int slot) const;
    void clearTexture(const std::string &name, int slot) const;

protected:
    GLuint createShader(std::string version, std::vector<std::string> extensions, std::vector<std::string> defines,
                        const char* shaderData, const GLint shaderSize, ShaderType type);
    void checkCompileErrors(GLuint shader, ShaderType type);

    template <typename T>
    bool isUniformCached(const std::string &name, const T &value) const;

protected:
    mutable std::unordered_map<std::string, std::any> uniformCache;
    static GLuint bindedShaderID;
};

} // namespace quasar

#endif // SHADER_BASE_H
