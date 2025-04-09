#ifndef OPENGL_OBJECT_H
#define OPENGL_OBJECT_H

#include <Utils/Platform.h>

namespace quasar {

class OpenGLObject {
public:
    GLuint ID;

    operator GLuint() const { return ID; }

    OpenGLObject() : ID(0) {}
    virtual ~OpenGLObject() = default;

    virtual void bind() const = 0;
    virtual void unbind() const = 0;
};

} // namespace quasar

#endif // OPENGL_OBJECT_H
