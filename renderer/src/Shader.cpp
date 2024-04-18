#include <Shaders/Shader.h>

void Shader::loadFromFile(const std::string vertexPath, const std::string fragmentPath, const std::string geometryPath) {
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
        // open files
        vShaderFile.open(vertexPath);
        fShaderFile.open(fragmentPath);

        // read file's buffer contents into streams
        std::stringstream vShaderStream, fShaderStream;
        vShaderStream << vShaderFile.rdbuf();
        fShaderStream << fShaderFile.rdbuf();

        // close file handlers
        vShaderFile.close();
        fShaderFile.close();

        // convert stream into string
        vertexCode = vShaderStream.str();
        fragmentCode = fShaderStream.str();

        // if geometry shader path is present, also load a geometry shader
        if (geometryPath != "") {
            gShaderFile.open(geometryPath);
            std::stringstream gShaderStream;
            gShaderStream << gShaderFile.rdbuf();
            gShaderFile.close();
            geometryCode = gShaderStream.str();
        }
    }
    catch (std::ifstream::failure& e) {
        std::cout << "ERROR::SHADER::FILE_NOT_SUCCESSFULLY_READ: " << e.what() << std::endl;
    }

    const char* vShaderCode = vertexCode.c_str();
    const char* fShaderCode = fragmentCode.c_str();
    const char* gShaderCode = geometryPath != "" ? geometryCode.c_str() : nullptr;

    unsigned int vertexDataSize = vertexCode.size();
    unsigned int fragmentDataSize = fragmentCode.size();
    unsigned int geometryDataSize = geometryPath != "" ? geometryCode.size() : 0;

    createAndCompileProgram(vShaderCode, vertexDataSize, fShaderCode, fragmentDataSize, gShaderCode, geometryDataSize);
}

void Shader::loadFromData(const char* vertexData, const GLint vertexDataSize,
                          const char* fragmentData, const GLint fragmentDataSize,
                          const char* geometryData, const GLint geometryDataSize) {
    createAndCompileProgram(vertexData, vertexDataSize, fragmentData, fragmentDataSize, geometryData, geometryDataSize);
}

void Shader::createAndCompileProgram(const char* vertexData, const GLint vertexDataSize,
                                     const char* fragmentData, const GLint fragmentDataSize,
                                     const char* geometryData, const GLint geometryDataSize) {

    // compile vertex shader
    GLuint vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &vertexData, &vertexDataSize);
    glCompileShader(vertex);
    checkCompileErrors(vertex, ShaderType::VERTEX);

    // compile fragment shader
    GLuint fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fragmentData, &fragmentDataSize);
    glCompileShader(fragment);
    checkCompileErrors(fragment, ShaderType::FRAGMENT);

    // if geometry shader is given, compile geometry shader
    GLuint geometry;
    if (geometryData != nullptr) {
        geometry = glCreateShader(GL_GEOMETRY_SHADER);
        glShaderSource(geometry, 1, &geometryData, &geometryDataSize);
        glCompileShader(geometry);
        checkCompileErrors(geometry, ShaderType::GEOMETRY);
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

void Shader::checkCompileErrors(GLuint shader, ShaderType type) {
    GLint success;
    GLchar infoLog[1024];
    if (type != ShaderType::PROGRAM) {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 1024, NULL, infoLog);
            std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << static_cast<int>(type) << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
        }
    }
    else {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shader, 1024, NULL, infoLog);
            std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << static_cast<int>(type) << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
        }
    }
}
