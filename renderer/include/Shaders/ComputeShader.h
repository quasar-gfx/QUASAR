#ifndef COMPUTE_SHADER_H
#define COMPUTE_SHADER_H

#include <string>

#include <Shaders/ShaderBase.h>

struct ComputeShaderDataCreateParams {
    std::string version = "430 core";
    const char* computeCodeData = nullptr;
    unsigned int computeCodeSize = 0;
    std::vector<std::string> defines;
};

struct ComputeShaderFileCreateParams {
    std::string version = "430 core";
    std::string computeCodePath = "";
    std::vector<std::string> defines;
};

class ComputeShader : public ShaderBase {
public:
    std::string version = "430 core";
    std::vector<std::string> defines;

    explicit ComputeShader(const ComputeShaderDataCreateParams& params) : version(params.version), defines(params.defines) {
        loadFromData(params.computeCodeData, params.computeCodeSize);
    }
    explicit ComputeShader(const ComputeShaderFileCreateParams& params) : version(params.version), defines(params.defines) {
        loadFromFile(params.computeCodePath);
    }

    void loadFromFile(const std::string &computePath);
    void loadFromData(const char* computeCodeData, const GLint computeCodeSize);

    void dispatch(GLuint numGroupsX, GLuint numGroupsY, GLuint numGroupsZ);
    void memoryBarrier(GLbitfield barriers);

private:
    void createAndCompileProgram(const char* computeCodeData, const GLint computeCodeSize);
};

#endif // COMPUTE_SHADER_H
