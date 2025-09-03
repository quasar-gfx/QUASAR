#include <Utils/FileIO.h>
#include <Shaders/Shader.h>

using namespace quasar;

Shader::Shader(const ShaderFileCreateParams& params)
    : version(params.version)
    , extensions(params.extensions)
    , defines(params.defines)
{
    loadFromFiles(params.vertexCodePath, params.fragmentCodePath, params.geometryCodePath);
}

Shader::Shader(const ShaderDataCreateParams& params)
    : version(params.version)
    , extensions(params.extensions)
    , defines(params.defines)
{
    loadFromData(params.vertexCodeData, params.vertexCodeSize, params.fragmentCodeData, params.fragmentCodeSize, params.geometryData, params.geometryMetadataSize);
}

void Shader::loadFromFiles(const std::string vertexPath, const std::string fragmentPath, const std::string geometryPath) {
    std::string vertexCode = FileIO::loadFromTextFile(vertexPath);
    std::string fragmentCode = FileIO::loadFromTextFile(fragmentPath);

    // If geometry shader path is present, also load a geometry shader
    std::string geometryCode;
    if (geometryPath != "") {
        geometryCode = FileIO::loadFromTextFile(geometryPath);
    }

    const char* vShaderCode = vertexCode.c_str();
    const char* fShaderCode = fragmentCode.c_str();
    const char* gShaderCode = geometryPath != "" ? geometryCode.c_str() : nullptr;

    size_t vertexCodeSize = vertexCode.size();
    size_t fragmentCodeSize = fragmentCode.size();
    size_t geometryMetadataSize = geometryPath != "" ? geometryCode.size() : 0;

    loadFromData(vShaderCode, vertexCodeSize, fShaderCode, fragmentCodeSize, gShaderCode, geometryMetadataSize);
}

void Shader::loadFromData(const char* vertexCodeData, const size_t vertexCodeSize,
                          const char* fragmentCodeData, const size_t fragmentCodeSize,
                          const char* geometryData, const size_t geometryMetadataSize) {
    createAndCompileProgram(vertexCodeData, vertexCodeSize, fragmentCodeData, fragmentCodeSize, geometryData, geometryMetadataSize);
}

void Shader::createAndCompileProgram(const char* vertexCodeData, const size_t vertexCodeSize,
                                     const char* fragmentCodeData, const size_t fragmentCodeSize,
                                     const char* geometryData, const size_t geometryMetadataSize) {
    // Compile vertex shader
    GLuint vertex = createShader(version, extensions, defines, vertexCodeData, vertexCodeSize, ShaderType::VERTEX);

    // Compile fragment shader
    GLuint fragment = createShader(version, extensions, defines, fragmentCodeData, fragmentCodeSize, ShaderType::FRAGMENT);

    // If geometry shader is given, compile geometry shader
    GLuint geometry;
    if (geometryData != nullptr) {
        geometry = createShader(version, extensions, defines, geometryData, geometryMetadataSize, ShaderType::GEOMETRY);
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
