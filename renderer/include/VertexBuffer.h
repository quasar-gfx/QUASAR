#ifndef VERTEX_BUFFER_H
#define VERTEX_BUFFER_H

#include <glad/glad.h>

#include <OpenGLObject.h>

class VertexArray : public OpenGLObject {
public:
    VertexArray() {
        glGenVertexArrays(1, &ID);
    }

    ~VertexArray() {
        cleanup();
    }

    void bind() {
        glBindVertexArray(ID);
    }

    void unbind() {
        glBindVertexArray(0);
    }

    void cleanup() {
        glDeleteVertexArrays(1, &ID);
    }
};

class VertexBuffer : public OpenGLObject {
public:
    VertexArray va;

    VertexBuffer(const void* data, unsigned int size) : va() {
        glGenBuffers(1, &ID);
        va.bind();
        glBindBuffer(GL_ARRAY_BUFFER, ID);
        glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW);
    }

    ~VertexBuffer() {
        cleanup();
    }

    void bind() {
        va.bind();
    }

    void unbind() {
        glBindVertexArray(0);
    }

    void cleanup() {
        glDeleteBuffers(1, &ID);
        va.cleanup();
    }

    void addAttribute(unsigned int index, unsigned int size, GLboolean normalized, unsigned int stride, size_t offset) {
        glEnableVertexAttribArray(index);
        glVertexAttribPointer(index, size, GL_FLOAT, normalized, stride, (const void*)offset);
    }
};

#endif // VERTEX_BUFFER_H
