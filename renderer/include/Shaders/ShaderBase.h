#ifndef SHADER_BASE_H
#define SHADER_BASE_H

#include <vector>

#include <glad/glad.h>

#include <glm/glm.hpp>

#include <OpenGLObject.h>

enum class ShaderType {
    PROGRAM,
    VERTEX,
    FRAGMENT,
    GEOMETRY,
    COMPUTE
};

class ShaderBase : public OpenGLObject {
public:
    explicit ShaderBase() = default;

    void bind() {
        glUseProgram(ID);
    }

    void unbind() {
        glUseProgram(0);
    }

    void cleanup() {
        glDeleteProgram(ID);
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

protected:
    GLuint createShader(std::string version, std::vector<std::string> defines, const char* shaderData, const GLint shaderSize, ShaderType type) {
        std::string versionStr = "#version " + version + "\n";

        std::vector<std::string> definesWithNewline;
        for (const auto& define : defines) {
            definesWithNewline.push_back(define + "\n");
        }

        std::vector<GLchar const*> shaderSrcs;
        shaderSrcs.push_back(versionStr.c_str());
        for (const auto& define : definesWithNewline) {
            shaderSrcs.push_back(define.c_str());
        }
        shaderSrcs.push_back(shaderData);

        std::vector<GLint> shaderSrcsSizes;
        shaderSrcsSizes.push_back(versionStr.size());
        for (const auto& define : definesWithNewline) {
            shaderSrcsSizes.push_back(define.size());
        }
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
};

#endif // SHADER_BASE_H
