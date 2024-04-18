#ifndef SHADER_H
#define SHADER_H

#include <glad/glad.h>

#include <glm/glm.hpp>

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include <OpenGLObject.h>

#include <shaders.h>

enum class ShaderType {
    PROGRAM,
    VERTEX,
    FRAGMENT,
    GEOMETRY,
    COMPUTE
};

struct ShaderCreateParams {
    std::string vertexCodePath = "";
    std::string fragmentCodePath = "";
    std::string geometryCodePath = "";
    const char* vertexCodeData = nullptr;
    unsigned int vertexCodeSize = 0;
    const char* fragmentCodeData = nullptr;
    unsigned int fragmentCodeSize = 0;
    const char* geometryData = nullptr;
    unsigned int geometryDataSize = 0;
};

class Shader : public OpenGLObject {
public:
    explicit Shader() = default;

    explicit Shader(const ShaderCreateParams& params) {
        if (params.vertexCodeData != nullptr && params.fragmentCodeData != nullptr) {
            loadFromData(params.vertexCodeData, params.vertexCodeSize, params.fragmentCodeData, params.fragmentCodeSize, params.geometryData, params.geometryDataSize);
        }
        else {
            loadFromFile(params.vertexCodePath, params.fragmentCodeData, params.geometryCodePath);
        }
    }

    void loadFromFile(const std::string vertexPath, const std::string fragmentPath, const std::string geometryPath = "");
    void loadFromData(const char* vertexCodeData, const GLint vertexCodeSize,
                      const char* fragmentCodeData, const GLint fragmentCodeSize,
                      const char* geometryData = nullptr, const GLint geometryDataSize = 0);

    ~Shader() {
        cleanup();
    }

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

private:
    void createAndCompileProgram(const char* vertexCodeData, const GLint vertexCodeSize,
                                 const char* fragmentCodeData, const GLint fragmentCodeSize,
                                 const char* geometryData = nullptr, const GLint geometryDataSize = 0);

protected:
    void checkCompileErrors(GLuint shader, ShaderType type);
};

#endif // SHADER_H
