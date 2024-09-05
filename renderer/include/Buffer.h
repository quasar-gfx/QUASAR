#ifndef BUFFER_H
#define BUFFER_H

#include <iostream>
#include <type_traits>

#include <vector>
#include <OpenGLObject.h>

template<typename T>
struct Buffer : OpenGLObject {
public:
    GLenum target;
    GLenum usage;

    Buffer(GLenum target = GL_ARRAY_BUFFER, GLenum usage = GL_STATIC_DRAW)
        : target(target), usage(usage), numElems(0) {
        glGenBuffers(1, &ID);
    }

    Buffer(GLenum target, GLenum usage, unsigned int numElems, const T* data)
        : Buffer(target, usage) {
        bind();
        setData(numElems, data);
    }

    ~Buffer() override {
        glDeleteBuffers(1, &ID);
    }

    // Copy constructor
    Buffer(const Buffer<T>& other)
        : target(other.target), usage(other.usage), numElems(other.numElems) {
        glGenBuffers(1, &ID);
        bind();
        std::vector<T> data(other.numElems);
        other.getSubData(0, other.numElems, data.data());
        setData(data);
        unbind();
    }

    // Copy assignment operator
    Buffer<T>& operator=(const Buffer<T>& other) {
        if (this == &other) {
            return *this;
        }

        // Delete existing buffer
        glDeleteBuffers(1, &ID);

        // Recreate the buffer and copy data
        target = other.target;
        usage = other.usage;
        numElems = other.numElems;
        glGenBuffers(1, &ID);

        if (numElems > 0) {
            bind();
            std::vector<T> data(numElems);
            other.getSubData(0, numElems, data.data());
            setData(data);
            unbind();
        }

        return *this;
    }

    // Move constructor
    Buffer(Buffer<T>&& other) noexcept
        : target(other.target), usage(other.usage), numElems(other.numElems) {
        ID = other.ID;
        other.ID = 0;
        other.numElems = 0;
    }

    // Move assignment operator
    Buffer<T>& operator=(Buffer<T>&& other) noexcept {
        if (this == &other) {
            return *this;
        }

        // Delete current buffer
        glDeleteBuffers(1, &ID);

        // Move the data
        ID = other.ID;
        target = other.target;
        usage = other.usage;
        numElems = other.numElems;

        other.ID = 0;
        other.numElems = 0;

        return *this;
    }

    unsigned int getSize() const {
        return numElems;
    }

    void setSize(unsigned int numElems) {
        this->numElems = numElems;
    }

    void resize(unsigned int numElems) {
        setSize(numElems);
        glBufferData(target, numElems * sizeof(T), nullptr, usage);
    }

    void setData(unsigned int numElems, const void* data) {
        setSize(numElems);
        glBufferData(target, numElems * sizeof(T), data, usage);
    }

    void setData(const std::vector<T>& data) {
        setData(data.size(), data.data());
    }

    void setSubData(unsigned int offset, unsigned int numElems, const void* data) {
        glBufferSubData(target, offset * sizeof(T), numElems * sizeof(T), data);
    }

    void setSubData(unsigned int offset, const std::vector<T>& data) {
        setSubData(offset, data.size(), data.data());
    }

#ifdef GL_CORE
    void getSubData(unsigned int offset, unsigned int numElems, void* data) const {
        glGetBufferSubData(target, offset * sizeof(T), numElems * sizeof(T), data);
    }

    std::vector<T> getData() const {
        std::vector<T> data(numElems);
        bind();

        T* mappedBuffer = static_cast<T*>(glMapBuffer(target, GL_READ_ONLY));
        if (mappedBuffer) {
            std::copy(mappedBuffer, mappedBuffer + numElems, data.begin());

            glUnmapBuffer(target);
        } else {
            std::cerr << "Error: Could not map buffer data." << std::endl;
        }

        unbind();
        return data;
    }
#endif

    void bind() const override {
        glBindBuffer(target, ID);
    }

    void unbind() const override {
        glBindBuffer(target, 0);
    }

private:
    unsigned int numElems;
};

#endif // BUFFER_H
