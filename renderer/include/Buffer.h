#ifndef BUFFER_H
#define BUFFER_H

#include <type_traits>

#include <OpenGLObject.h>

template<typename T>
struct Buffer : OpenGLObject {
public:
    GLenum type;
    GLenum usage;

    Buffer(GLenum type = GL_ARRAY_BUFFER, GLenum usage = GL_STATIC_DRAW)
            : type(type)
            , usage(usage)
            , numElems(0) {
        glGenBuffers(1, &ID);
    }
    ~Buffer() override {
        glDeleteBuffers(1, &ID);
    }

    unsigned int getSize() const {
        return numElems;
    }

    void setSize(unsigned int numElems) {
        this->numElems = numElems;
    }

    void resize(unsigned int numElems) {
        setSize(numElems);
        glBufferData(type, numElems * sizeof(T), nullptr, usage);
    }

    void setData(unsigned int numElems, const void *data) {
        setSize(numElems);
        glBufferData(type, numElems * sizeof(T), data, usage);
    }

    void setData(const std::vector<T> &data) {
        setData(data.size(), data.data());
    }

    void bind() const override {
        glBindBuffer(type, ID);
    }

    void unbind() const override {
        glBindBuffer(type, 0);
    }

private:
    unsigned int numElems;
};

#endif // BUFFER_H
