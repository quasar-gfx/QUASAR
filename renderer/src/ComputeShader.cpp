#include <fstream>
#include <sstream>
#include <iostream>

#include <Shaders/ComputeShader.h>

void ComputeShader::loadFromFile(std::string computePath) {
    std::ifstream computeFile;

    computeFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    try {
        computeFile.open(computePath);

        std::stringstream computeStream;
        computeStream << computeFile.rdbuf();
        computeFile.close();

        std::string computeCode = computeStream.str();
        const char* computeCodeCStr = computeCode.c_str();
        unsigned int computeCodeSize = computeCode.size();
        loadFromData(computeCodeCStr, computeCodeSize);
    }
    catch (std::ifstream::failure e) {
        std::cerr << "Failed to read compute shader file: " << computePath << std::endl;
    }
}

void ComputeShader::loadFromData(const char* computeCodeData, const GLint computeCodeSize) {
    createAndCompileProgram(computeCodeData, computeCodeSize);
}

void ComputeShader::createAndCompileProgram(const char* computeCodeData, const GLint computeCodeSize) {
    GLuint compute = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(compute, 1, &computeCodeData, &computeCodeSize);
    glCompileShader(compute);
    checkCompileErrors(compute, ShaderType::COMPUTE);

    ID = glCreateProgram();
    glAttachShader(ID, compute);
    glLinkProgram(ID);
    checkCompileErrors(ID, ShaderType::PROGRAM);

    glDeleteShader(compute);
}

void ComputeShader::dispatch(GLuint numGroupsX, GLuint numGroupsY, GLuint numGroupsZ) {
    glDispatchCompute(numGroupsX, numGroupsY, numGroupsZ);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}
