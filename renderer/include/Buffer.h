#ifndef BUFFER_H
#define BUFFER_H

#include <OpenGLObject.h>

struct Buffer : OpenGLObject {
    GLenum type;
    unsigned int size;

    Buffer(GLenum type = GL_ARRAY_BUFFER) : type(type), size(0) {
        glGenBuffers(1, &ID);
    }
    ~Buffer() override {
        glDeleteBuffers(1, &ID);
    }

    void setSize(unsigned int size) {
        this->size = size;
    }

    void setData(unsigned int size, const void *data, GLenum usage = GL_STATIC_DRAW) {
        setSize(size);
        glBufferData(type, size, data, usage);
    }

    void bind() const override {
        glBindBuffer(type, ID);
    }

    void unbind() const override {
        glBindBuffer(type, 0);
    }
};

#endif // BUFFER_H
