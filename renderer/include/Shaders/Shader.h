#ifndef SHADER_H
#define SHADER_H

#include <string>

#include <Shaders/ShaderBase.h>

// Pre-stored shaders
#include <shaders_builtin.h>

namespace quasar {

struct ShaderDataCreateParams {
#ifdef GL_CORE
    std::string version = "410 core";
#else
    std::string version = "310 es";
#endif
    const char* vertexCodeData = nullptr;
    uint vertexCodeSize = 0;
    const char* fragmentCodeData = nullptr;
    uint fragmentCodeSize = 0;
    const char* geometryData = nullptr;
    uint geometryDataSize = 0;
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

    Shader() = default;
    Shader(const ShaderFileCreateParams& params)
            : version(params.version)
            , extensions(params.extensions)
            , defines(params.defines) {
        loadFromFiles(params.vertexCodePath, params.fragmentCodePath, params.geometryCodePath);
    }
    Shader(const ShaderDataCreateParams& params)
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

} // namespace quasar

#endif // SHADER_H
