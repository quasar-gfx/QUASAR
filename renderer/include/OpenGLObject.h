#ifndef OPENGL_OBJECT_H
#define OPENGL_OBJECT_H

#include <glad/glad.h>

class OpenGLObject {
public:
    GLuint ID;

    operator GLuint() const { return ID; }

    explicit OpenGLObject() : ID(0) {}
    ~OpenGLObject() {};

    virtual void bind() = 0;
    virtual void unbind() = 0;
};

#endif // OPENGL_OBJECT_H
