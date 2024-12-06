#ifndef COMPUTE_SHADER_H
#define COMPUTE_SHADER_H

#include <string>

#include <Buffer.h>
#include <Texture.h>
#include <Shaders/ShaderBase.h>

struct ComputeShaderDataCreateParams {
#ifdef GL_CORE
    std::string version = "430 core";
#else
    std::string version = "320 es";
#endif
    const char* computeCodeData = nullptr;
    unsigned int computeCodeSize = 0;
    std::vector<std::string> extensions;
    std::vector<std::string> defines;
};

struct ComputeShaderFileCreateParams {
#ifdef GL_CORE
    std::string version = "430 core";
#else
    std::string version = "320 es";
#endif
    std::string computeCodePath = "";
    std::vector<std::string> extensions;
    std::vector<std::string> defines;
};

class ComputeShader : public ShaderBase {
public:
    std::string version = "430 core";
    std::vector<std::string> extensions;
    std::vector<std::string> defines;

    ComputeShader(const ComputeShaderDataCreateParams& params)
            : version(params.version)
            , extensions(params.extensions)
            , defines(params.defines) {
        loadFromData(params.computeCodeData, params.computeCodeSize);
    }
    ComputeShader(const ComputeShaderFileCreateParams& params)
            : version(params.version)
            , extensions(params.extensions)
            , defines(params.defines) {
        loadFromFile(params.computeCodePath);
    }

    template <template<typename> class Buffer, typename T>
    void setBuffer(GLenum target, int slot, const Buffer<T>& buffer) const {
        glBindBufferBase(target, slot, buffer);
    }
    void clearBuffer(GLenum target, int slot) const {
        glBindBufferBase(target, slot, 0);
    }

    void setImageTexture(int slot, const Texture& texture, GLuint level, GLboolean layered, GLuint layer, GLenum access, GLenum format) {
        glBindImageTexture(slot, texture, level, layered, layer, access, format);
    }

    void loadFromFile(const std::string &computePath);
    void loadFromData(const char* computeCodeData, const GLint computeCodeSize);

    void dispatch(GLuint numGroupsX, GLuint numGroupsY, GLuint numGroupsZ);
    void memoryBarrier(GLbitfield barriers);

    void startTiming();
    void endTiming();
    double getElapsedTime() const;

private:
    void createAndCompileProgram(const char* computeCodeData, const GLint computeCodeSize);

    mutable GLuint startQueryID = 0;
    mutable GLuint endQueryID = 0;
    mutable bool isQueried = false;
    mutable GLuint64 lastElapsedTime = 0;
};

#endif // COMPUTE_SHADER_H
