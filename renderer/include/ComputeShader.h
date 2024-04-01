#ifndef COMPUTE_SHADER_H
#define COMPUTE_SHADER_H

#include <glad/glad.h>

#include <string>

#include <Shader.h>

struct ComputeShaderCreateParams {
    std::string computeCodePath = "";
    const char* computeData = nullptr;
};

class ComputeShader : public Shader {
public:
    explicit ComputeShader(const ComputeShaderCreateParams& params) {
        if (params.computeData != nullptr) {
            loadFromData(params.computeData);
        }
        else {
            loadFromFile(params.computeCodePath);
        }
    }

    void loadFromFile(std::string computePath);
    void loadFromData(const char* computeData);

    void dispatch(GLuint numGroupsX, GLuint numGroupsY, GLuint numGroupsZ);

    ~ComputeShader() {
        cleanup();
    }

private:
    void createAndCompileProgram(const char* computeData);
};

#endif // COMPUTE_SHADER_H
