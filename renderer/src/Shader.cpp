#include <Shaders/Shader.h>

void Shader::loadFromFiles(const std::string vertexPath, const std::string fragmentPath, const std::string geometryPath) {
    std::string vertexCode;
    std::string fragmentCode;
    std::string geometryCode;
    std::ifstream vShaderFile;
    std::ifstream fShaderFile;
    std::ifstream gShaderFile;

    // ensure ifstream objects can throw exceptions
    vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    gShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    try {
        // load vertex shader
        vShaderFile.open(vertexPath);

        std::stringstream vShaderStream;
        vShaderStream << vShaderFile.rdbuf();
        vShaderFile.close();

        vertexCode = vShaderStream.str();

        // load fragment shader
        fShaderFile.open(fragmentPath);

        std::stringstream fShaderStream;
        fShaderStream << fShaderFile.rdbuf();
        fShaderFile.close();

        fragmentCode = fShaderStream.str();

        // if geometry shader path is present, also load a geometry shader
        if (geometryPath != "") {
            gShaderFile.open(geometryPath);

            std::stringstream gShaderStream;
            gShaderStream << gShaderFile.rdbuf();
            gShaderFile.close();

            geometryCode = gShaderStream.str();
        }

        const char* vShaderCode = vertexCode.c_str();
        const char* fShaderCode = fragmentCode.c_str();
        const char* gShaderCode = geometryPath != "" ? geometryCode.c_str() : nullptr;

        unsigned int vertexCodeSize = vertexCode.size();
        unsigned int fragmentCodeSize = fragmentCode.size();
        unsigned int geometryDataSize = geometryPath != "" ? geometryCode.size() : 0;

        loadFromData(vShaderCode, vertexCodeSize, fShaderCode, fragmentCodeSize, gShaderCode, geometryDataSize);
    }
    catch (std::ifstream::failure& e) {
        std::cerr << "Failed to read shader files: " << e.what() << std::endl;
    }
}

void Shader::loadFromData(const char* vertexCodeData, const GLint vertexCodeSize,
                          const char* fragmentCodeData, const GLint fragmentCodeSize,
                          const char* geometryData, const GLint geometryDataSize) {
    createAndCompileProgram(vertexCodeData, vertexCodeSize, fragmentCodeData, fragmentCodeSize, geometryData, geometryDataSize);
}

void Shader::createAndCompileProgram(const char* vertexCodeData, const GLint vertexCodeSize,
                                     const char* fragmentCodeData, const GLint fragmentCodeSize,
                                     const char* geometryData, const GLint geometryDataSize) {
    // compile vertex shader
    GLuint vertex = createShader(version, defines, vertexCodeData, vertexCodeSize, ShaderType::VERTEX);

    // compile fragment shader
    GLuint fragment = createShader(version, defines, fragmentCodeData, fragmentCodeSize, ShaderType::FRAGMENT);

    // if geometry shader is given, compile geometry shader
    GLuint geometry;
    if (geometryData != nullptr) {
        geometry = createShader(version, defines, geometryData, geometryDataSize, ShaderType::GEOMETRY);
    }

    // shader Program
    ID = glCreateProgram();
    glAttachShader(ID, vertex);
    glAttachShader(ID, fragment);
    if (geometryData != nullptr) {
        glAttachShader(ID, geometry);
    }

    glLinkProgram(ID);
    checkCompileErrors(ID, ShaderType::PROGRAM);

    // delete the shaders as they're linked into our program now and no longer necessary
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    if (geometryData != nullptr) {
        glDeleteShader(geometry);
    }
}
