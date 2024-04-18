#ifndef COMPUTE_SHADER_H
#define COMPUTE_SHADER_H

#include <glad/glad.h>

#include <string>

#include <Shaders/Shader.h>

struct ComputeShaderCreateParams {
    std::string computeCodePath = "";
    const char* computeData = nullptr;
    unsigned int computeDataSize = 0;
};

class ComputeShader : public Shader {
public:
    explicit ComputeShader(const ComputeShaderCreateParams& params) {
        if (params.computeData != nullptr) {
            loadFromData(params.computeData, params.computeDataSize);
        }
        else {
            loadFromFile(params.computeCodePath);
        }
    }

    void loadFromFile(std::string computePath);
    void loadFromData(const char* computeData, const GLint computeDataSize);

    void dispatch(GLuint numGroupsX, GLuint numGroupsY, GLuint numGroupsZ);

    ~ComputeShader() {
        cleanup();
    }

private:
    void createAndCompileProgram(const char* computeData, const GLint computeDataSize);
};

#endif // COMPUTE_SHADER_H
