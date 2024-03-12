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
        loadFromData(computeCodeCStr);
    }
    catch (std::ifstream::failure e) {
        std::cerr << "Failed to read compute shader file: " << computePath << std::endl;
    }
}

void ComputeShader::loadFromData(const char* computeData) {
    createAndCompileProgram(computeData);
}

void ComputeShader::createAndCompileProgram(const char* computeData) {
    GLuint compute = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(compute, 1, &computeData, NULL);
    glCompileShader(compute);
    checkCompileErrors(compute, SHADER_COMPUTE);

    ID = glCreateProgram();
    glAttachShader(ID, compute);
    glLinkProgram(ID);
    checkCompileErrors(ID, SHADER_PROGRAM);

    glDeleteShader(compute);
}
