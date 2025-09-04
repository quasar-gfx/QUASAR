#include <regex>
#include <sstream>
#include <iomanip>
#include <spdlog/spdlog.h>
#include <Shaders/ShaderBase.h>

using namespace quasar;

GLuint ShaderBase::bindedShaderID = 0;

ShaderBase::~ShaderBase() {
    glDeleteProgram(ID);
}

void ShaderBase::bind() const {
    if (bindedShaderID == ID) {
        return;
    }

    glUseProgram(ID);
    bindedShaderID = ID;
}

void ShaderBase::unbind() const {
    if (bindedShaderID == 0) {
        return;
    }

    glUseProgram(0);
    bindedShaderID = 0;
}

void ShaderBase::setBool(const std::string& name, bool value) const {
    if (!isUniformCached(name, value)) {
        glUniform1i(glGetUniformLocation(ID, name.c_str()), (int)value);
    }
}

void ShaderBase::setUint(const std::string& name, uint value) const {
    if (!isUniformCached(name, value)) {
        glUniform1ui(glGetUniformLocation(ID, name.c_str()), value);
    }
}

void ShaderBase::setInt(const std::string& name, int value) const {
    if (!isUniformCached(name, value)) {
        glUniform1i(glGetUniformLocation(ID, name.c_str()), value);
        uniformCache[name] = value;
    }
}

void ShaderBase::setFloat(const std::string& name, float value) const {
    if (!isUniformCached(name, value)) {
        glUniform1f(glGetUniformLocation(ID, name.c_str()), value);
        uniformCache[name] = value;
    }
}

void ShaderBase::setVec2(const std::string& name, const glm::vec2& value) const {
    if (!isUniformCached(name, value)) {
        glUniform2fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
        uniformCache[name] = value;
    }
}

void ShaderBase::setVec3(const std::string& name, const glm::vec3& value) const {
    if (!isUniformCached(name, value)) {
        glUniform3fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
        uniformCache[name] = value;
    }
}

void ShaderBase::setVec4(const std::string& name, const glm::vec4& value) const {
    if (!isUniformCached(name, value)) {
        glUniform4fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
        uniformCache[name] = value;
    }
}

void ShaderBase::setMat2(const std::string& name, const glm::mat2& mat) const {
    if (!isUniformCached(name, mat)) {
        glUniformMatrix2fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
        uniformCache[name] = mat;
    }
}

void ShaderBase::setMat3(const std::string& name, const glm::mat3& mat) const {
    if (!isUniformCached(name, mat)) {
        glUniformMatrix3fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
        uniformCache[name] = mat;
    }
}

void ShaderBase::setMat4(const std::string& name, const glm::mat4& mat) const {
    if (!isUniformCached(name, mat)) {
        glUniformMatrix4fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
        uniformCache[name] = mat;
    }
}

void ShaderBase::setTexture(const Texture& texture, int slot) const {
    texture.bind(slot);
}

void ShaderBase::setTexture(const std::string& name, const Texture& texture, int slot) const {
    texture.bind(slot);
    if (!isUniformCached(name, slot)) {
        setInt(name, slot);
        uniformCache[name] = slot;
    }
}

void ShaderBase::clearTexture(const std::string& name, int slot) const {
    glBindTexture(GL_TEXTURE_2D, 0);
    if (!isUniformCached(name, slot)) {
        setInt(name, slot);
        uniformCache[name] = 0;
    }
}

GLuint ShaderBase::createShader(std::string version, std::vector<std::string> extensions, std::vector<std::string> defines,
                                const char* shaderData, const GLint shaderSize, ShaderType type) {
    std::vector<std::string> sources;
    shaderSourceLines.clear();

    // Add version string
    sources.emplace_back("#version " + version + "\n");

    // Extensions
#if defined(PLATFORM_OPENXR)
    sources.emplace_back("#extension GL_OVR_multiview : enable\n");
#endif
#ifdef GL_ES
    sources.emplace_back("#extension GL_EXT_shader_io_blocks : enable\n");
#endif
    for (const auto& ext : extensions) {
        sources.emplace_back(ext + "\n");
    }

    // Platform defines
#if defined(PLATFORM_ANDROID)
    sources.emplace_back("#define ANDROID\n");
#elif defined(PLATFORM_LINUX)
    sources.emplace_back("#define LINUX\n");
#elif defined(PLATFORM_APPLE)
    sources.emplace_back("#define APPLE\n");
#elif defined(PLATFORM_WINDOWS)
    sources.emplace_back("#define WINDOWS\n");
#endif

    // Precision defines
    sources.emplace_back("precision highp float;\n");
    sources.emplace_back("precision highp int;\n");
    sources.emplace_back("precision highp sampler2D;\n");

    // User defines
    for (const auto& def : defines) {
        sources.emplace_back(def + "\n");
    }

    // Add shader source code
    std::istringstream shaderStream(std::string(shaderData, shaderSize));
    std::string line;
    while (std::getline(shaderStream, line)) {
        sources.push_back(line + "\n");
    }

    shaderSourceLines = sources;

    std::vector<const GLchar*> cSources;
    std::vector<GLint> lengths;
    for (const auto& src : sources) {
        cSources.push_back(src.c_str());
        lengths.push_back((GLint)src.size());
    }

    GLuint shader;
    switch (type) {
        case ShaderType::VERTEX:
            shader = glCreateShader(GL_VERTEX_SHADER);
            break;
        case ShaderType::FRAGMENT:
            shader = glCreateShader(GL_FRAGMENT_SHADER);
            break;
        case ShaderType::GEOMETRY:
            shader = glCreateShader(GL_GEOMETRY_SHADER);
            break;
        case ShaderType::COMPUTE:
            shader = glCreateShader(GL_COMPUTE_SHADER);
            break;
        default:
            spdlog::error("Invalid shader type: {}", static_cast<int>(type));
            return -1;
    }

    glShaderSource(shader, cSources.size(), cSources.data(), lengths.data());
    glCompileShader(shader);
    checkCompileErrors(shader,type);

    return shader;
}

void ShaderBase::checkCompileErrors(GLuint shader, ShaderType type) {
    std::string shaderTypeStr;
    switch (type) {
        case ShaderType::VERTEX:   shaderTypeStr = "VERTEX"; break;
        case ShaderType::FRAGMENT: shaderTypeStr = "FRAGMENT"; break;
        case ShaderType::GEOMETRY: shaderTypeStr = "GEOMETRY"; break;
        case ShaderType::COMPUTE:  shaderTypeStr = "COMPUTE"; break;
        default:                   shaderTypeStr = "UNKNOWN"; break;
    }

    GLint success;
    GLchar infoLog[1024];
    if (type != ShaderType::PROGRAM) {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, sizeof(infoLog), nullptr, infoLog);

            std::string infoStr(infoLog);
            spdlog::error("Failed to compile {} shader:\n{}", shaderTypeStr, infoStr);

            // Try to extract line number
#if defined(GL_CORE) && !defined(__APPLE__)
            // GL Core error: "0(111) : message"
            std::regex lineRegex(R"(\((\d+)\))");
#else
            // GLES error: "ERROR: 0:111: message"
            std::regex lineRegex(R"(ERROR:\s*\d+:(\d+):)");
#endif
            std::smatch match;
            int errorLine = -1;
            if (std::regex_search(infoStr, match, lineRegex)) {
                errorLine = std::stoi(match[1]);
            }

            // Show context if we know the line and have source
            if (errorLine > 0 && !shaderSourceLines.empty()) {
                int lineIndex = errorLine - 1;
                int startLine = std::max(0, lineIndex - 5);
                int endLine = std::min((int)shaderSourceLines.size(), lineIndex + 6);

                std::stringstream contextStream;
                contextStream << "---- Shader Error Context ----\n";
                for (int i = startLine; i < endLine; i++) {
                    if (i == lineIndex) {
                        contextStream << ">>> ";
                    }
                    else {
                        contextStream << "    ";
                    }
                    contextStream << std::setw(4) << i + 1 << ": " << shaderSourceLines[i];
                }

                spdlog::error("{}", contextStream.str());
            }
            else {
                spdlog::warn("Could not extract line number or shader source is empty.");
            }
        }
    }
    else {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shader, sizeof(infoLog), nullptr, infoLog);
            spdlog::error("Failed to link program:\n{}", infoLog);
        }
    }
}

template <typename T>
bool ShaderBase::isUniformCached(const std::string& name, const T& value) const {
    auto it = uniformCache.find(name);
    if (it != uniformCache.end()) {
        try {
            const T& cachedValue = std::any_cast<const T&>(it->second);
            return cachedValue == value;
        } catch (const std::bad_any_cast&) {
            return false;
        }
    }
    return false;
}
