#include <Utils/FileIO.h>
#include <Utils/TimeUtils.h>

#include <Shaders/ComputeShader.h>

using namespace quasar;

ComputeShader::ComputeShader(const ComputeShaderDataCreateParams& params)
    : version(params.version)
    , extensions(params.extensions)
    , defines(params.defines)
{
    loadFromData(params.computeCodeData, params.computeCodeSize);
}

ComputeShader::ComputeShader(const ComputeShaderFileCreateParams& params)
    : version(params.version)
    , extensions(params.extensions)
    , defines(params.defines)
{
    loadFromFile(params.computeCodePath);
}

void ComputeShader::startTiming() {
#ifdef GL_CORE
    if (!startQueryID) {
        glGenQueries(1, &startQueryID);
    }
    glQueryCounter(startQueryID, GL_TIMESTAMP);
    isQueried = true;
#else
    startTime = timeutils::getTimeNanos();
#endif
}

void ComputeShader::endTiming() {
#ifdef GL_CORE
    if (!endQueryID) {
        glGenQueries(1, &endQueryID);
    }
    glQueryCounter(endQueryID, GL_TIMESTAMP);

    isQueried = false;
#else
    endTime = timeutils::getTimeNanos();
    lastElapsedTime = endTime - startTime;
#endif
}

double ComputeShader::getElapsedTime() const {
#ifdef GL_CORE
    if (isQueried) {
        return timeutils::nanosToMillis(lastElapsedTime);
    }

    if (startQueryID && endQueryID) {
        GLuint64 startTime = 0, endTime = 0;

        glGetQueryObjectui64v(startQueryID, GL_QUERY_RESULT, &startTime);
        glGetQueryObjectui64v(endQueryID, GL_QUERY_RESULT, &endTime);

        lastElapsedTime = endTime - startTime;
        isQueried = true;
    }
#endif

    return timeutils::nanosToMillis(lastElapsedTime);
}

void ComputeShader::setBuffer(GLenum target, int slot, const Buffer& buffer) const {
    glBindBufferBase(target, slot, buffer);
}
void ComputeShader::clearBuffer(GLenum target, int slot) const {
    glBindBufferBase(target, slot, 0);
}

void ComputeShader::setImageTexture(int slot, const Texture& texture, GLuint level, GLboolean layered, GLuint layer, GLenum access, GLenum format) const {
    glBindImageTexture(slot, texture, level, layered, layer, access, format);
}

void ComputeShader::loadFromFile(const std::string& computePath) {
    std::string computeCode = FileIO::loadFromTextFile(computePath);

    const char* cShaderCode = computeCode.c_str();
    size_t computeCodeSize = computeCode.size();

    loadFromData(cShaderCode, computeCodeSize);
}

void ComputeShader::loadFromData(const char* computeCodeData, const size_t computeCodeSize) {
    createAndCompileProgram(computeCodeData, computeCodeSize);
}

void ComputeShader::createAndCompileProgram(const char* computeCodeData, const size_t computeCodeSize) {
    GLuint compute = createShader(version, extensions, defines, computeCodeData, computeCodeSize, ShaderType::COMPUTE);

    ID = glCreateProgram();
    glAttachShader(ID, compute);

    glLinkProgram(ID);
    checkCompileErrors(ID, ShaderType::PROGRAM);

    glDeleteShader(compute);
}

void ComputeShader::dispatch(GLuint numGroupsX, GLuint numGroupsY, GLuint numGroupsZ) const {
    glDispatchCompute(numGroupsX, numGroupsY, numGroupsZ);
}

void ComputeShader::memoryBarrier(GLbitfield barriers) const {
    glMemoryBarrier(barriers);
}
