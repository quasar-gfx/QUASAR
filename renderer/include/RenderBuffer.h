#ifndef RENDERBUFFER_H
#define RENDERBUFFER_H

#include <glad/glad.h>

#include <OpenGLObject.h>

class RenderBuffer : public OpenGLObject {
public:
    unsigned int width, height;

    RenderBuffer(unsigned int width, unsigned int height, GLenum internalFormat = GL_DEPTH_COMPONENT24) : width(width), height(height) {
        glGenRenderbuffers(1, &ID);
        glBindRenderbuffer(GL_RENDERBUFFER, ID);
        glRenderbufferStorage(GL_RENDERBUFFER, internalFormat, width, height);
    }

    ~RenderBuffer() {
        cleanup();
    }

    void bind() {
        bind(0);
    }

    void bind(unsigned int slot = 0) {
        glActiveTexture(GL_TEXTURE0 + slot);
        glBindRenderbuffer(GL_RENDERBUFFER, ID);
    }

    void unbind() {
        glBindTexture(GL_TEXTURE_2D, 0);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);
    }

    void cleanup() {
        glDeleteRenderbuffers(1, &ID);
    }
};

#endif // RENDERBUFFER_H
