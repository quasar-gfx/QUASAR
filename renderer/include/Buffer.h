#ifndef BUFFER_H
#define BUFFER_H

#include <iostream>
#include <cstring>
#include <vector>

#include <OpenGLObject.h>

template<typename T>
struct Buffer : OpenGLObject {
public:
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

    // copy constructor
    Buffer(const Buffer<T>& other)
            : target(other.target), usage(other.usage), numElems(other.numElems) {
        glGenBuffers(1, &ID);
        bind();
        std::vector<T> data(other.numElems);
        other.getSubData(0, other.numElems, data.data());
        setData(data);
        unbind();
    }

    // copy assignment operator
    Buffer<T>& operator=(const Buffer<T>& other) {
        if (this == &other) {
            return *this;
        }
        glDeleteBuffers(1, &ID);

        // recreate the buffer and copy data
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

    // move constructor
    Buffer(Buffer<T>&& other) noexcept
            : target(other.target), usage(other.usage), numElems(other.numElems) {
        ID = other.ID;
        other.ID = 0;
        other.numElems = 0;
    }

    // move assignment operator
    Buffer<T>& operator=(Buffer<T>&& other) noexcept {
        if (this == &other) {
            return *this;
        }

        glDeleteBuffers(1, &ID);

        // move the data
        ID = other.ID;
        target = other.target;
        usage = other.usage;
        numElems = other.numElems;

        other.ID = 0;
        other.numElems = 0;

        return *this;
    }

    void bind() const override {
        glBindBuffer(target, ID);
    }

    void unbind() const override {
        glBindBuffer(target, 0);
    }

    unsigned int getSize() const {
        return numElems;
    }

    void resize(unsigned int newNumElems, bool copy = false) {
        if (numElems == newNumElems) {
            return;
        }

        std::vector<T> data;
        if (copy) {
            data.resize(numElems);
            getData(data.data());
        }

        glBufferData(target, newNumElems * sizeof(T), nullptr, usage);

        if (copy) {
            unsigned int elemsToCopy = std::min(numElems, newNumElems);
            glBufferSubData(target, 0, elemsToCopy * sizeof(T), data.data());
        }

        numElems = newNumElems;
    }

    void smartResize(unsigned int newNumElems, bool copy = false) {
        if (newNumElems > numElems) {
            resize(numElems * 2, copy);
        }
        else if (newNumElems <= numElems / 4) {
            resize(numElems / 4, copy);
            std::cout << "Resizing buffer to " << numElems / 4 << " elements." << std::endl;
        }
    }

#ifdef GL_CORE
    void getSubData(unsigned int offset, unsigned int numElems, void* data) const {
        glGetBufferSubData(target, offset * sizeof(T), numElems * sizeof(T), data);
    }
#endif

    void getData(void* data) const {
#ifdef GL_CORE
        // getSubData is faster than mapping the buffer
        getSubData(0, numElems, data);
#else
        T* mappedBuffer = static_cast<T*>(glMapBufferRange(target, 0, numElems * sizeof(T), GL_MAP_READ_BIT));
        if (mappedBuffer) {
            std::memcpy(data, mappedBuffer, numElems * sizeof(T));

            glUnmapBuffer(target);
        } else {
            std::cerr << "Error: Could not map buffer data." << std::endl;
        }
#endif
    }

    std::vector<T> getData() const {
        std::vector<T> data(numElems);
        getData(data.data());
        return data;
    }

    void setData(unsigned int numElems, const void* data) {
        resize(numElems);
        glBufferData(target, numElems * sizeof(T), data, usage);
    }

    void setData(const std::vector<T>& data) {
        setData(data.size(), data.data());
    }

#ifdef GL_CORE
    void setSubData(unsigned int offset, unsigned int numElems, const void* data) {
        glBufferSubData(target, offset * sizeof(T), numElems * sizeof(T), data);
    }
#endif

    void setSubData(unsigned int offset, const std::vector<T>& data) {
        setSubData(offset, data.size(), data.data());
    }

private:
    GLenum target;
    GLenum usage;

    unsigned int numElems;
};

#endif // BUFFER_H
