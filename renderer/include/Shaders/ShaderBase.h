#ifndef SHADER_BASE_H
#define SHADER_BASE_H

#include <any>
#include <unordered_map>
#include <iostream>
#include <vector>

#include <glm/glm.hpp>

#include <OpenGLObject.h>
#include <Texture.h>

enum class ShaderType {
    PROGRAM,
    VERTEX,
    FRAGMENT,
    GEOMETRY,
    COMPUTE
};

class ShaderBase : public OpenGLObject {
public:
    ShaderBase() = default;
    ~ShaderBase() override {
        glDeleteProgram(ID);
    }

    void bind() const override {
        if (bindedShaderID == ID) {
            return;
        }

        glUseProgram(ID);
        bindedShaderID = ID;
    }

    void unbind() const override {
        if (bindedShaderID == 0) {
            return;
        }

        glUseProgram(0);
        bindedShaderID = 0;
    }

    void setBool(const std::string &name, bool value) const {
        if (!isUniformCached(name, value)) {
            glUniform1i(glGetUniformLocation(ID, name.c_str()), (int)value);
        }
    }

    void setInt(const std::string &name, int value) const {
        if (!isUniformCached(name, value)) {
            glUniform1i(glGetUniformLocation(ID, name.c_str()), value);
            uniformCache[name] = value;
        }
    }

    void setFloat(const std::string &name, float value) const {
        if (!isUniformCached(name, value)) {
            glUniform1f(glGetUniformLocation(ID, name.c_str()), value);
            uniformCache[name] = value;
        }
    }

    void setVec2(const std::string &name, const glm::vec2 &value) const {
        if (!isUniformCached(name, value)) {
            glUniform2fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
            uniformCache[name] = value;
        }
    }

    void setVec3(const std::string &name, const glm::vec3 &value) const {
        if (!isUniformCached(name, value)) {
            glUniform3fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
            uniformCache[name] = value;
        }
    }

    void setVec4(const std::string &name, const glm::vec4 &value) const {
        if (!isUniformCached(name, value)) {
            glUniform4fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
            uniformCache[name] = value;
        }
    }

    void setMat2(const std::string &name, const glm::mat2 &mat) const {
        if (!isUniformCached(name, mat)) {
            glUniformMatrix2fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
            uniformCache[name] = mat;
        }
    }

    void setMat3(const std::string &name, const glm::mat3 &mat) const {
        if (!isUniformCached(name, mat)) {
            glUniformMatrix3fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
            uniformCache[name] = mat;
        }
    }

    void setMat4(const std::string &name, const glm::mat4 &mat) const {
        if (!isUniformCached(name, mat)) {
            glUniformMatrix4fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
            uniformCache[name] = mat;
        }
    }

    void setTexture(const Texture &texture, int slot) const {
        texture.bind(slot);
    }

    void setTexture(const std::string &name, const Texture &texture, int slot) const {
        texture.bind(slot);
        if (!isUniformCached(name, slot)) {
            setInt(name, slot);
            uniformCache[name] = slot;
        }
    }

    void clearTexture(const std::string &name, int slot) const {
        glBindTexture(GL_TEXTURE_2D, 0);
        if (!isUniformCached(name, slot)) {
            setInt(name, slot);
            uniformCache[name] = 0;
        }
    }

protected:
    GLuint createShader(std::string version, std::vector<std::string> extensions, std::vector<std::string> defines,
                        const char* shaderData, const GLint shaderSize, ShaderType type) {
        std::vector<GLchar const*> shaderSrcs;
        std::vector<GLint> shaderSrcsSizes;

        // add version string
        std::string versionStr = "#version " + version + "\n";
        shaderSrcs.push_back(versionStr.c_str());
        shaderSrcsSizes.push_back(versionStr.size());

        std::vector<std::string> extensionsWithNewline;
#ifdef GL_ES
        extensionsWithNewline.push_back("#extension GL_EXT_shader_io_blocks : enable\n");
#endif
#if defined(__ANDROID__)
        extensionsWithNewline.push_back("#extension GL_OVR_multiview : enable\n");
#endif
        // user extensions
        for (const auto& extension : extensions) {
            extensionsWithNewline.push_back(extension + "\n");
        }

        // add extensions
        for (const auto& extension : extensionsWithNewline) {
            shaderSrcs.push_back(extension.c_str());
            shaderSrcsSizes.push_back(extension.size());
        }

        std::vector<std::string> definesWithNewline;
        // add platform defines
#ifdef GL_ES
        definesWithNewline.push_back("#define PLATFORM_ES\n");
#endif
#if defined(__ANDROID__)
        definesWithNewline.push_back("#define ANDROID\n");
#elif defined(_WIN32) || defined(_WIN64)
        definesWithNewline.push_back("#define WINDOWS\n");
        definesWithNewline.push_back("#define PLATFORM_CORE\n");
#elif defined(__linux__)
        definesWithNewline.push_back("#define LINUX\n");
        definesWithNewline.push_back("#define PLATFORM_CORE\n");
#elif defined(__APPLE__)
        definesWithNewline.push_back("#define APPLE\n");
        definesWithNewline.push_back("#define PLATFORM_CORE\n");
#endif
        // add precision defines
        definesWithNewline.push_back("precision highp float;\n");
        definesWithNewline.push_back("precision highp int;\n");
        definesWithNewline.push_back("precision highp sampler2D;\n");

        // user defines
        for (const auto& define : defines) {
            definesWithNewline.push_back(define + "\n");
        }

        // add defines
        for (const auto& define : definesWithNewline) {
            shaderSrcs.push_back(define.c_str());
            shaderSrcsSizes.push_back(define.size());
        }

        // add shader data
        shaderSrcs.push_back(shaderData);
        shaderSrcsSizes.push_back(shaderSize);

        GLuint shader;
        switch (type) {
            case ShaderType::VERTEX:
                shader = glCreateShader(GL_VERTEX_SHADER);
                break;
            case ShaderType::FRAGMENT:
                shader = glCreateShader(GL_FRAGMENT_SHADER);
                break;
            case ShaderType::GEOMETRY:
                shader = glCreateShader(GL_GEOMETRY_SHADER);
                break;
            case ShaderType::COMPUTE:
                shader = glCreateShader(GL_COMPUTE_SHADER);
                break;
            default:
                std::cerr << "Invalid shader type: " << static_cast<int>(type) << std::endl;
                return -1;
        }

        glShaderSource(shader, shaderSrcs.size(), shaderSrcs.data(), shaderSrcsSizes.data());
        glCompileShader(shader);
        checkCompileErrors(shader, type);

        return shader;
    }

    void checkCompileErrors(GLuint shader, ShaderType type) {
        GLint success;
        GLchar infoLog[1024];
        if (type != ShaderType::PROGRAM) {
            glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
            if (!success) {
                glGetShaderInfoLog(shader, 1024, NULL, infoLog);
                std::cerr << "Failed to compile shader of type: " << static_cast<int>(type) << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
            }
        }
        else {
            glGetProgramiv(shader, GL_LINK_STATUS, &success);
            if (!success) {
                glGetProgramInfoLog(shader, 1024, NULL, infoLog);
                std::cerr << "Failed to link shader of type: " << static_cast<int>(type) << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
            }
        }
    }

protected:
    mutable std::unordered_map<std::string, std::any> uniformCache;

    template <typename T>
    bool isUniformCached(const std::string &name, const T &value) const {
        auto it = uniformCache.find(name);
        if (it != uniformCache.end()) {
            try {
                const T& cachedValue = std::any_cast<const T&>(it->second);
                return cachedValue == value;
            } catch (const std::bad_any_cast&) {
                return false;
            }
        }

        return false;
    }

    static GLuint bindedShaderID;
};

#endif // SHADER_BASE_H
