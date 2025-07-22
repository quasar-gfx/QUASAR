#ifndef BUFFER_H
#define BUFFER_H

#include <vector>
#include <cstring>
#include <type_traits>

#include <spdlog/spdlog.h>

#include <OpenGLObject.h>

namespace quasar {

class Buffer : public OpenGLObject {
public:
    Buffer() {
        glGenBuffers(1, &ID);
    }
    Buffer(GLenum target, size_t dataSize, GLenum usage = GL_STATIC_DRAW)
        : target(target)
        , numElems(0)
        , dataSize(dataSize)
        , usage(usage)
    {
        glGenBuffers(1, &ID);
    }
    Buffer(GLenum target, uint numElems, size_t dataSize, const void* data, GLenum usage = GL_STATIC_DRAW)
        : target(target)
        , numElems(numElems)
        , dataSize(dataSize)
        , usage(usage)
    {
        glGenBuffers(1, &ID);
        bind();
        setData(numElems, data);
    }
    Buffer(const Buffer& other)
        : target(other.target)
        , usage(other.usage)
        , numElems(other.numElems)
        , dataSize(other.dataSize)
    {
        glGenBuffers(1, &ID);
        bind();
        std::vector<char> data(other.numElems * other.dataSize);
#ifdef GL_CORE
        other.getSubData(0, other.numElems, data.data());
#else
        other.getData(data.data());
#endif
        setData(other.numElems, data.data());
        unbind();
    }
    Buffer(Buffer&& other) noexcept
        : target(other.target)
        , usage(other.usage)
        , numElems(other.numElems)
        , dataSize(other.dataSize)
    {
        other.ID = 0;
        other.numElems = 0;
    }
    ~Buffer() override {
        glDeleteBuffers(1, &ID);
    }

    Buffer& operator=(const Buffer& other) {
        if (this == &other) return *this;

        glDeleteBuffers(1, &ID);

        target = other.target;
        usage = other.usage;
        numElems = other.numElems;
        dataSize = other.dataSize;
        glGenBuffers(1, &ID);

        if (numElems > 0) {
            bind();
            std::vector<char> data(numElems * dataSize);
#ifdef GL_CORE
            other.getSubData(0, numElems, data.data());
#else
            other.getData(data.data());
#endif
            setData(numElems, data.data());
            unbind();
        }

        return *this;
    }

    Buffer& operator=(Buffer&& other) noexcept {
        if (this == &other) return *this;

        glDeleteBuffers(1, &ID);

        ID = other.ID;
        target = other.target;
        usage = other.usage;
        numElems = other.numElems;
        dataSize = other.dataSize;

        other.ID = 0;
        other.numElems = 0;

        return *this;
    }

    void bind() const override {
        glBindBuffer(target, ID);
    }

    void bindToUniformBlock(GLuint shaderID, const std::string& blockName, GLuint bindingIndex) const {
        GLuint blockIndex = glGetUniformBlockIndex(shaderID, blockName.c_str());
        if (blockIndex == GL_INVALID_INDEX) {
            return;
        }
        glUniformBlockBinding(shaderID, blockIndex, bindingIndex);
        glBindBufferBase(target, bindingIndex, ID);
    }

    void unbind() const override {
        glBindBuffer(target, 0);
    }

    uint getSize() const {
        return numElems;
    }

    void resize(uint newNumElems, bool copy = false) {
        if (numElems == newNumElems) return;

        std::vector<char> data;
        if (copy) {
            data.resize(numElems * dataSize);
            getData(data.data());
        }

        glBufferData(target, newNumElems * dataSize, nullptr, usage);

        if (copy) {
            uint elemsToCopy = std::min(numElems, newNumElems);
            glBufferSubData(target, 0, elemsToCopy * dataSize, data.data());
        }

        numElems = newNumElems;
    }

    void smartResize(uint newNumElems, bool copy = false) {
        if (newNumElems > numElems) {
            resize(numElems * 2, copy);
        }
        else if (newNumElems <= numElems / 4) {
            resize(numElems / 4, copy);
        }
    }

#ifdef GL_CORE
    void getSubData(uint offset, uint numElems, void* data) const {
        glGetBufferSubData(target, offset * dataSize, numElems * dataSize, data);
    }
#endif

    void getData(void* data) const {
#ifdef GL_CORE
        getSubData(0, numElems, data);
#else
        void* mappedBuffer = glMapBufferRange(target, 0, numElems * dataSize, GL_MAP_READ_BIT);
        if (mappedBuffer) {
            std::memcpy(data, mappedBuffer, numElems * dataSize);
            glUnmapBuffer(target);
        }
        else {
            spdlog::error("Could not map buffer data.");
        }
#endif
    }

    template<typename T>
    std::vector<T> getData() const {
        static_assert(std::is_trivially_copyable<T>::value, "Buffer data must be trivially copyable.");

        if (sizeof(T) != dataSize) {
            spdlog::error("Data size mismatch. Requested type has size {}, but buffer holds size {}.", sizeof(T), dataSize);
            return {};
        }

        std::vector<T> data(numElems);
        getData(static_cast<void*>(data.data()));
        return data;
    }

    void setData(uint numElems, const void* data) {
        resize(numElems);
        glBufferData(target, numElems * dataSize, data, usage);
    }

    void setData(const std::vector<char>& data) {
        setData(data.size() / dataSize, data.data());
    }

#ifdef GL_CORE
    void setSubData(uint offset, uint numElems, const void* data) {
        glBufferSubData(target, offset * dataSize, numElems * dataSize, data);
    }

    void setSubData(uint offset, const std::vector<char>& data) {
        setSubData(offset, data.size() / dataSize, data.data());
    }
#endif

private:
    GLenum target = GL_ARRAY_BUFFER;
    GLenum usage = GL_STATIC_DRAW;
    uint numElems;
    size_t dataSize;
};

} // namespace quasar

#endif // BUFFER_H
