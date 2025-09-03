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
    size_t vertexCodeSize = 0;
    const char* fragmentCodeData = nullptr;
    size_t fragmentCodeSize = 0;
    const char* geometryData = nullptr;
    size_t geometryMetadataSize = 0;
    std::vector<std::string> extensions;
    std::vector<std::string> defines;
};

struct ShaderFileCreateParams {
#ifdef GL_CORE
    std::string version = "410 core";
#else
    std::string version = "320 es";
#endif
    std::string vertexCodePath = "";
    std::string fragmentCodePath = "";
    std::string geometryCodePath = "";
    std::vector<std::string> extensions;
    std::vector<std::string> defines;
};

class Shader : public ShaderBase {
public:
    std::string version;
    std::vector<std::string> extensions;
    std::vector<std::string> defines;

    Shader() = default;
    Shader(const ShaderFileCreateParams& params);
    Shader(const ShaderDataCreateParams& params);

    void loadFromFiles(const std::string vertexPath, const std::string fragmentPath, const std::string geometryPath = "");
    void loadFromData(const char* vertexCodeData, const size_t vertexCodeSize,
                      const char* fragmentCodeData, const size_t fragmentCodeSize,
                      const char* geometryData = nullptr, const size_t geometryMetadataSize = 0);

private:
    void createAndCompileProgram(const char* vertexCodeData, const size_t vertexCodeSize,
                                 const char* fragmentCodeData, const size_t fragmentCodeSize,
                                 const char* geometryData = nullptr, const size_t geometryMetadataSize = 0);
};

} // namespace quasar

#endif // SHADER_H
