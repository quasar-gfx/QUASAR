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
        std::cout << "ERROR::SHADER::FILE_NOT_SUCCESSFULLY_READ: " << e.what() << std::endl;
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
    std::string versionStr = "#version " + version + "\n";

    // compile vertex shader
    std::vector<GLchar const*> vertexFiles = { versionStr.c_str(), vertexCodeData };
    std::vector<GLint> vertexFilesSizes   = { static_cast<GLint>(versionStr.size()), vertexCodeSize };
    GLuint vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, vertexFiles.size(), vertexFiles.data(), vertexFilesSizes.data());
    glCompileShader(vertex);
    checkCompileErrors(vertex, ShaderType::VERTEX);

    // compile fragment shader
    std::vector<GLchar const*> fragmentFiles = { versionStr.c_str(), fragmentCodeData };
    std::vector<GLint> fragmentFilesSizes   = { static_cast<GLint>(versionStr.size()), fragmentCodeSize };
    GLuint fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, fragmentFiles.size(), fragmentFiles.data(), fragmentFilesSizes.data());
    glCompileShader(fragment);
    checkCompileErrors(fragment, ShaderType::FRAGMENT);

    // if geometry shader is given, compile geometry shader
    GLuint geometry;
    if (geometryData != nullptr) {
        std::vector<GLchar const*> geometryFiles = { versionStr.c_str(), geometryData };
        std::vector<GLint> geometryFilesSizes   = { static_cast<GLint>(versionStr.size()), geometryDataSize };
        geometry = glCreateShader(GL_GEOMETRY_SHADER);
        glShaderSource(geometry, geometryFiles.size(), geometryFiles.data(), geometryFilesSizes.data());
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
