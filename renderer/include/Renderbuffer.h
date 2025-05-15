#ifndef RENDERBUFFER_H
#define RENDERBUFFER_H

#include <OpenGLObject.h>

namespace quasar {

struct RenderbufferCreateParams {
    uint width = 0;
    uint height = 0;
    GLenum internalFormat = GL_DEPTH_COMPONENT24;
};

class Renderbuffer : public OpenGLObject {
public:
    uint width, height;

    GLint internalFormat = GL_RGB;

    Renderbuffer(const RenderbufferCreateParams& params)
            : width(params.width)
            , height(params.height)
            , internalFormat(params.internalFormat)
            , OpenGLObject() {
        glGenRenderbuffers(1, &ID);
        glBindRenderbuffer(GL_RENDERBUFFER, ID);
        glRenderbufferStorage(GL_RENDERBUFFER, internalFormat, width, height);
    }
    ~Renderbuffer() override {
        glDeleteRenderbuffers(1, &ID);
    }

    void resize(uint width, uint height, GLenum internalFormat = GL_DEPTH_COMPONENT24) {
        this->width = width;
        this->height = height;
        glBindRenderbuffer(GL_RENDERBUFFER, ID);
        glRenderbufferStorage(GL_RENDERBUFFER, internalFormat, width, height);
    }

    void bind() const override {
        glBindRenderbuffer(GL_RENDERBUFFER, ID);
    }

    void unbind() const override {
        glBindRenderbuffer(GL_RENDERBUFFER, 0);
    }
};

} // namespace quasar

#endif // RENDERBUFFER_H
