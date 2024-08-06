#ifndef SHADER_H
#define SHADER_H

#include <string>

#include <Shaders/ShaderBase.h>

// pre-stored shaders
#include <shaders.h>

struct ShaderDataCreateParams {
#ifdef GL_CORE
    std::string version = "410 core";
#else
    std::string version = "310 es";
#endif
    const char* vertexCodeData = nullptr;
    unsigned int vertexCodeSize = 0;
    const char* fragmentCodeData = nullptr;
    unsigned int fragmentCodeSize = 0;
    const char* geometryData = nullptr;
    unsigned int geometryDataSize = 0;
    std::vector<std::string> extensions;
    std::vector<std::string> defines;
};

struct ShaderFileCreateParams {
#ifdef GL_CORE
    std::string version = "410 core";
#else
    std::string version = "310 es";
#endif
    std::string vertexCodePath = "";
    std::string fragmentCodePath = "";
    std::string geometryCodePath = "";
    std::vector<std::string> extensions;
    std::vector<std::string> defines;
};

class Shader : public ShaderBase {
public:
    std::string version = "410 core";
    std::vector<std::string> extensions;
    std::vector<std::string> defines;

    explicit Shader() = default;
    explicit Shader(const ShaderFileCreateParams& params)
            : version(params.version)
            , extensions(params.extensions)
            , defines(params.defines) {
        loadFromFiles(params.vertexCodePath, params.fragmentCodePath, params.geometryCodePath);
    }
    explicit Shader(const ShaderDataCreateParams& params)
            : version(params.version)
            , extensions(params.extensions)
            , defines(params.defines) {
        loadFromData(params.vertexCodeData, params.vertexCodeSize, params.fragmentCodeData, params.fragmentCodeSize, params.geometryData, params.geometryDataSize);
    }

    void loadFromFiles(const std::string vertexPath, const std::string fragmentPath, const std::string geometryPath = "");
    void loadFromData(const char* vertexCodeData, const GLint vertexCodeSize,
                      const char* fragmentCodeData, const GLint fragmentCodeSize,
                      const char* geometryData = nullptr, const GLint geometryDataSize = 0);

private:
    void createAndCompileProgram(const char* vertexCodeData, const GLint vertexCodeSize,
                                 const char* fragmentCodeData, const GLint fragmentCodeSize,
                                 const char* geometryData = nullptr, const GLint geometryDataSize = 0);
};

#endif // SHADER_H
