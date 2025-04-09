#include <Utils/FileIO.h>
#include <Utils/TimeUtils.h>

#include <Shaders/ComputeShader.h>

using namespace quasar;

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
        return timeutils::nanoToMillis(lastElapsedTime);
    }

    if (startQueryID && endQueryID) {
        GLuint64 startTime = 0, endTime = 0;

        glGetQueryObjectui64v(startQueryID, GL_QUERY_RESULT, &startTime);
        glGetQueryObjectui64v(endQueryID, GL_QUERY_RESULT, &endTime);

        lastElapsedTime = endTime - startTime;
        isQueried = true;
    }
#endif

    return timeutils::nanoToMillis(lastElapsedTime);
}

void ComputeShader::loadFromFile(const std::string &computePath) {
    std::string computeCode = FileIO::loadTextFile(computePath);

    const char* cShaderCode = computeCode.c_str();
    unsigned int computeCodeSize = computeCode.size();

    loadFromData(cShaderCode, computeCodeSize);
}

void ComputeShader::loadFromData(const char* computeCodeData, const GLint computeCodeSize) {
    createAndCompileProgram(computeCodeData, computeCodeSize);
}

void ComputeShader::createAndCompileProgram(const char* computeCodeData, const GLint computeCodeSize) {
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
