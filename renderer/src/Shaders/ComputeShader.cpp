#include <Utils/FileIO.h>
#include <Shaders/ComputeShader.h>

void ComputeShader::loadFromFile(const std::string &computePath) {

    std::string computeCode = FileIO::loadTextFile(computePath);

    const char* cShaderCode = computeCode.c_str();
    unsigned int computeCodeSize = computeCode.size();

    loadFromData(cShaderCode, computeCodeSize);
}

void ComputeShader::loadFromData(const char* computeCodeData, const GLint computeCodeSize) {
    createAndCompileProgram(computeCodeData, computeCodeSize);
}

void ComputeShader::createAndCompileProgram(const char* computeCodeData, const GLint computeCodeSize) {
    GLuint compute = createShader(version, extensions, defines, computeCodeData, computeCodeSize, ShaderType::COMPUTE);

    ID = glCreateProgram();
    glAttachShader(ID, compute);
    glLinkProgram(ID);
    checkCompileErrors(ID, ShaderType::PROGRAM);

    glDeleteShader(compute);
}

void ComputeShader::dispatch(GLuint numGroupsX, GLuint numGroupsY, GLuint numGroupsZ) const {
    glDispatchCompute(numGroupsX, numGroupsY, numGroupsZ);
}

void ComputeShader::memoryBarrier(GLbitfield barriers) const {
    glMemoryBarrier(barriers);
}
