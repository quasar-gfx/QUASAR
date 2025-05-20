#include <Utils/FileIO.h>
#include <Shaders/Shader.h>

using namespace quasar;

void Shader::loadFromFiles(const std::string vertexPath, const std::string fragmentPath, const std::string geometryPath) {
    std::string vertexCode = FileIO::loadTextFile(vertexPath);
    std::string fragmentCode = FileIO::loadTextFile(fragmentPath);

    // If geometry shader path is present, also load a geometry shader
    std::string geometryCode;
    if (geometryPath != "") {
        geometryCode = FileIO::loadTextFile(geometryPath);
    }

    const char* vShaderCode = vertexCode.c_str();
    const char* fShaderCode = fragmentCode.c_str();
    const char* gShaderCode = geometryPath != "" ? geometryCode.c_str() : nullptr;

    uint vertexCodeSize = vertexCode.size();
    uint fragmentCodeSize = fragmentCode.size();
    uint geometryDataSize = geometryPath != "" ? geometryCode.size() : 0;

    loadFromData(vShaderCode, vertexCodeSize, fShaderCode, fragmentCodeSize, gShaderCode, geometryDataSize);
}

void Shader::loadFromData(const char* vertexCodeData, const GLint vertexCodeSize,
                          const char* fragmentCodeData, const GLint fragmentCodeSize,
                          const char* geometryData, const GLint geometryDataSize) {
    createAndCompileProgram(vertexCodeData, vertexCodeSize, fragmentCodeData, fragmentCodeSize, geometryData, geometryDataSize);
}

void Shader::createAndCompileProgram(const char* vertexCodeData, const GLint vertexCodeSize,
                                     const char* fragmentCodeData, const GLint fragmentCodeSize,
                                     const char* geometryData, const GLint geometryDataSize) {
    // Compile vertex shader
    GLuint vertex = createShader(version, extensions, defines, vertexCodeData, vertexCodeSize, ShaderType::VERTEX);

    // Compile fragment shader
    GLuint fragment = createShader(version, extensions, defines, fragmentCodeData, fragmentCodeSize, ShaderType::FRAGMENT);

    // If geometry shader is given, compile geometry shader
    GLuint geometry;
    if (geometryData != nullptr) {
        geometry = createShader(version, extensions, defines, geometryData, geometryDataSize, ShaderType::GEOMETRY);
    }

    // Shader Program
    ID = glCreateProgram();
    glAttachShader(ID, vertex);
    glAttachShader(ID, fragment);
    if (geometryData != nullptr) {
        glAttachShader(ID, geometry);
    }

    glLinkProgram(ID);
    checkCompileErrors(ID, ShaderType::PROGRAM);

    // Delete the shaders as they're linked into our program now and no longer necessary
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    if (geometryData != nullptr) {
        glDeleteShader(geometry);
    }
}
