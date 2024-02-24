#ifndef OPENGL_OBJECT_H
#define OPENGL_OBJECT_H

#include "glad/glad.h"

class OpenGLObject {
public:
    GLuint ID;

    OpenGLObject() : ID(0) {}
    ~OpenGLObject() {};

    virtual void bind() = 0;
    virtual void unbind() = 0;
    virtual void cleanup() = 0;
};

#endif // OPENGL_OBJECT_H
