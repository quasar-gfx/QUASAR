#ifndef SHADER_BASE_H
#define SHADER_BASE_H

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
        if (currentShaderID == ID) {
            return;
        }

        glUseProgram(ID);
        currentShaderID = ID;
    }

    void unbind() const override {
        if (currentShaderID == 0) {
            return;
        }

        glUseProgram(0);
        currentShaderID = 0;
    }

    void setBool(const std::string &name, bool value) const {
        glUniform1i(glGetUniformLocation(ID, name.c_str()), (int)value);
    }

    void setInt(const std::string &name, int value) const {
        glUniform1i(glGetUniformLocation(ID, name.c_str()), value);
    }

    void setFloat(const std::string &name, float value) const {
        glUniform1f(glGetUniformLocation(ID, name.c_str()), value);
    }

    void setVec2(const std::string &name, const glm::vec2 &value) const {
        glUniform2fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
    }
    void setVec2(const std::string &name, float x, float y) const {
        glUniform2f(glGetUniformLocation(ID, name.c_str()), x, y);
    }

    void setVec3(const std::string &name, const glm::vec3 &value) const {
        glUniform3fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
    }
    void setVec3(const std::string &name, float x, float y, float z) const {
        glUniform3f(glGetUniformLocation(ID, name.c_str()), x, y, z);
    }

    void setVec4(const std::string &name, const glm::vec4 &value) const {
        glUniform4fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
    }
    void setVec4(const std::string &name, float x, float y, float z, float w) {
        glUniform4f(glGetUniformLocation(ID, name.c_str()), x, y, z, w);
    }

    void setMat2(const std::string &name, const glm::mat2 &mat) const {
        glUniformMatrix2fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
    }

    void setMat3(const std::string &name, const glm::mat3 &mat) const {
        glUniformMatrix3fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
    }

    void setMat4(const std::string &name, const glm::mat4 &mat) const {
        glUniformMatrix4fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
    }

    void setTexture(const Texture &texture, int slot) const {
        texture.bind(slot);
    }

    void setTexture(const std::string &name, const Texture &texture, int slot) const {
        texture.bind(slot);
        setInt(name, slot);
    }

    void setTextureToEmpty(const std::string &name, int slot) const {
        glBindTexture(GL_TEXTURE_2D, 0);
        setInt(name, slot);
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
#ifdef GL_ES
        definesWithNewline.push_back("precision mediump float;\n");
        definesWithNewline.push_back("#define PLATFORM_ES\n");
#endif

        // add platform defines
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

    static GLuint currentShaderID;
};

#endif // SHADER_BASE_H
