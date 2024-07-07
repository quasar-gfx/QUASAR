#ifndef RENDERBUFFER_H
#define RENDERBUFFER_H

#include <glad/glad.h>

#include <OpenGLObject.h>

struct RenderbufferCreateParams {
    unsigned int width = 0;
    unsigned int height = 0;
    GLenum internalFormat = GL_DEPTH_COMPONENT24;
};

class Renderbuffer : public OpenGLObject {
public:
    unsigned int width, height;

    GLint internalFormat = GL_RGB;

    explicit Renderbuffer(const RenderbufferCreateParams &params)
            : width(params.width)
            , height(params.height)
            , internalFormat(params.internalFormat)
            , OpenGLObject() {
        glGenRenderbuffers(1, &ID);
        glBindRenderbuffer(GL_RENDERBUFFER, ID);
        glRenderbufferStorage(GL_RENDERBUFFER, internalFormat, width, height);
    }
    ~Renderbuffer() {
        glDeleteRenderbuffers(1, &ID);
    }

    void resize(unsigned int width, unsigned int height, GLenum internalFormat = GL_DEPTH_COMPONENT24) {
        this->width = width;
        this->height = height;
        glBindRenderbuffer(GL_RENDERBUFFER, ID);
        glRenderbufferStorage(GL_RENDERBUFFER, internalFormat, width, height);
    }

    void bind() const {
        glBindRenderbuffer(GL_RENDERBUFFER, ID);
    }

    void unbind() const {
        glBindRenderbuffer(GL_RENDERBUFFER, 0);
    }
};

#endif // RENDERBUFFER_H
