#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H

#include <glad/glad.h>

#include <OpenGLObject.h>
#include <Texture.h>
#include <CubeMap.h>

class Framebuffer : public OpenGLObject {
public:
    explicit Framebuffer() = default;

    ~Framebuffer() {
        cleanup();
    }

    void init() {
        glGenFramebuffers(1, &ID);
    }

    bool checkStatus(std::string name = "") {
        return glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
    }

    void attachTexture(const Texture &texture, GLenum attachment) {
        glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, texture.ID, 0);
    }

    void attachCubeMap(const CubeMap &cubeMap, GLenum attachment) {
        glFramebufferTexture(GL_FRAMEBUFFER, attachment, cubeMap.ID, 0);
    }

    void bind() {
        glBindFramebuffer(GL_FRAMEBUFFER, ID);
    }

    void unbind() {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void cleanup() {
        glDeleteFramebuffers(1, &ID);
    }
};

#endif // FRAMEBUFFER_H
