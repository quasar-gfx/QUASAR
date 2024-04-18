#include <fstream>
#include <sstream>
#include <iostream>

#include <ComputeShader.h>

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
        unsigned int computeDataSize = computeCode.size();
        loadFromData(computeCodeCStr, computeDataSize);
    }
    catch (std::ifstream::failure e) {
        std::cerr << "Failed to read compute shader file: " << computePath << std::endl;
    }
}

void ComputeShader::loadFromData(const char* computeData, const GLint computeDataSize) {
    createAndCompileProgram(computeData, computeDataSize);
}

void ComputeShader::createAndCompileProgram(const char* computeData, const GLint computeDataSize) {
    GLuint compute = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(compute, 1, &computeData, &computeDataSize);
    glCompileShader(compute);
    checkCompileErrors(compute, SHADER_COMPUTE);

    ID = glCreateProgram();
    glAttachShader(ID, compute);
    glLinkProgram(ID);
    checkCompileErrors(ID, SHADER_PROGRAM);

    glDeleteShader(compute);
}

void ComputeShader::dispatch(GLuint numGroupsX, GLuint numGroupsY, GLuint numGroupsZ) {
    glDispatchCompute(numGroupsX, numGroupsY, numGroupsZ);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}
