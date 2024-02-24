#ifndef INDEX_BUFFER_H
#define INDEX_BUFFER_H

#include "glad/glad.h"

#include "OpenGLObject.h"

class IndexBuffer : public OpenGLObject {
public:
    IndexBuffer() = default;

    IndexBuffer(const void* data, unsigned int size) {
        glGenBuffers(1, &ID);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ID);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, size * sizeof(unsigned int), data, GL_STATIC_DRAW);
    }

    ~IndexBuffer() {
        cleanup();
    }

    void bind() {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ID);
    }

    void unbind() {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    }

    void cleanup() {
        glDeleteBuffers(1, &ID);
    }
};

#endif // INDEX_BUFFER_H
