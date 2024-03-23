#ifndef COMPUTE_SHADER_H
#define COMPUTE_SHADER_H

#include <glad/glad.h>

#include <string>

#include <Shader.h>

class ComputeShader : public Shader {
public:
    void loadFromFile(std::string computePath);
    void loadFromData(const char* computeData);

    void dispatch(GLuint numGroupsX, GLuint numGroupsY, GLuint numGroupsZ);

    ~ComputeShader() {
        cleanup();
    }

private:
    void createAndCompileProgram(const char* computeData);
};

#endif // COMPUTE_SHADER_H
