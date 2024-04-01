#ifndef RENDERBUFFER_H
#define RENDERBUFFER_H

#include <glad/glad.h>

#include <OpenGLObject.h>

class Renderbuffer : public OpenGLObject {
public:
    unsigned int width, height;

    explicit Renderbuffer() = default;

    void create(unsigned int width, unsigned int height, GLenum internalFormat = GL_DEPTH_COMPONENT24) {
        this->width = width;
        this->height = height;

        glGenRenderbuffers(1, &ID);
        glBindRenderbuffer(GL_RENDERBUFFER, ID);
        glRenderbufferStorage(GL_RENDERBUFFER, internalFormat, width, height);
    }

    ~Renderbuffer() {
        cleanup();
    }

    void resize(unsigned int width, unsigned int height, GLenum internalFormat = GL_DEPTH_COMPONENT24) {
        this->width = width;
        this->height = height;
        glBindRenderbuffer(GL_RENDERBUFFER, ID);
        glRenderbufferStorage(GL_RENDERBUFFER, internalFormat, width, height);
    }

    void bind() {
        glBindRenderbuffer(GL_RENDERBUFFER, ID);
    }

    void unbind() {
        glBindRenderbuffer(GL_RENDERBUFFER, 0);
    }

    void cleanup() {
        glDeleteRenderbuffers(1, &ID);
    }
};

#endif // RENDERBUFFER_H
