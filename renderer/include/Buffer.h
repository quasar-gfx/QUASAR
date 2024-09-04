#ifndef BUFFER_H
#define BUFFER_H

#include <type_traits>

#include <OpenGLObject.h>

template<typename T>
struct Buffer : OpenGLObject {
public:
    GLenum target;
    GLenum usage;

    Buffer(GLenum target = GL_ARRAY_BUFFER, GLenum usage = GL_STATIC_DRAW)
            : target(target)
            , usage(usage)
            , numElems(0) {
        glGenBuffers(1, &ID);
    }
    Buffer(GLenum target, GLenum usage, unsigned int numElems, const T *data)
            : Buffer(target, usage) {
        bind();
        setData(numElems, data);
    }
    ~Buffer() override {
        glDeleteBuffers(1, &ID);
    }

    // copy constructor
    Buffer<T>& operator=(const Buffer<T>& other) {
        if (this == &other) {
            return *this;
        }

        ID = other.ID;
        target = other.target;
        usage = other.usage;
        numElems = other.numElems;

        // copy data from other buffer
        if (numElems > 0) {
            bind();
            std::vector<T> data(numElems);
            other.getSubData(0, numElems, data.data());
            setData(data);
            unbind();
        }

        return *this;
    }

    // move assignment
    Buffer<T>& operator=(Buffer<T>&& other) noexcept {
        if (this == &other) {
            return *this;
        }

        // delete current buffer so we can inherit the other buffer's ID
        glDeleteBuffers(1, &ID);

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

    void setData(unsigned int numElems, const void *data) {
        setSize(numElems);
        glBufferData(target, numElems * sizeof(T), data, usage);
    }

    void setData(const std::vector<T> &data) {
        setData(data.size(), data.data());
    }

    void setSubData(unsigned int offset, unsigned int numElems, const void *data) {
        glBufferSubData(target, offset * sizeof(T), numElems * sizeof(T), data);
    }

    void setSubData(unsigned int offset, const std::vector<T> &data) {
        setSubData(offset, data.size(), data.data());
    }

#ifdef GL_CORE
    void getSubData(unsigned int offset, unsigned int numElems, void *data) {
        glGetBufferSubData(target, offset * sizeof(T), numElems * sizeof(T), data);
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
