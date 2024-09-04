#ifndef BUFFER_H
#define BUFFER_H

#include <OpenGLObject.h>

struct Buffer : OpenGLObject {
public:
    GLenum type;
    GLenum usage;

    Buffer(GLenum type = GL_ARRAY_BUFFER, GLenum usage = GL_STATIC_DRAW)
            : type(type)
            , usage(usage)
            , size(0) {
        glGenBuffers(1, &ID);
    }
    ~Buffer() override {
        glDeleteBuffers(1, &ID);
    }

    unsigned int getSize() const {
        return size;
    }

    void setSize(unsigned int size) {
        this->size = size;
    }

    void setData(unsigned int size, const void *data) {
        setSize(size);
        glBufferData(type, size, data, usage);
    }

    void bind() const override {
        glBindBuffer(type, ID);
    }

    void unbind() const override {
        glBindBuffer(type, 0);
    }

private:
    unsigned int size;
};

#endif // BUFFER_H
