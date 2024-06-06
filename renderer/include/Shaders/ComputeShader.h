#ifndef COMPUTE_SHADER_H
#define COMPUTE_SHADER_H

#include <glad/glad.h>

#include <string>

#include <Shaders/Shader.h>

struct ComputeShaderCreateParams {
    std::string computeCodePath = "";
    const char* computeCodeData = nullptr;
    unsigned int computeCodeSize = 0;
};

class ComputeShader : public Shader {
public:
    explicit ComputeShader(const ComputeShaderCreateParams& params) {
        if (params.computeCodeData != nullptr) {
            loadFromData(params.computeCodeData, params.computeCodeSize);
        }
        else {
            loadFromFile(params.computeCodePath);
        }
    }

    void loadFromFile(const std::string &computePath);
    void loadFromData(const char* computeCodeData, const GLint computeCodeSize);

    void dispatch(GLuint numGroupsX, GLuint numGroupsY, GLuint numGroupsZ);

    ~ComputeShader() {
        cleanup();
    }

private:
    void createAndCompileProgram(const char* computeCodeData, const GLint computeCodeSize);
};

#endif // COMPUTE_SHADER_H
