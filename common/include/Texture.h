#ifndef TEXTURE_H
#define TEXTURE_H

#include <glad/glad.h>
#include <string>

#include <OpenGLObject.h>

enum TextureType {
    TEXTURE_DIFFUSE,
    TEXTURE_SPECULAR,
    TEXTURE_NORMAL,
    TEXTURE_HEIGHT
};

class Texture : public OpenGLObject {
public:
    GLuint ID;

    TextureType type = TEXTURE_DIFFUSE;

    unsigned int width, height;

    std::string path;

    void bind() {
        bind(0);
    }

    void bind(unsigned int slot) {
        glActiveTexture(GL_TEXTURE0 + slot);
        glBindTexture(GL_TEXTURE_2D, ID);
    }

    void unbind() {
        glBindTexture(GL_TEXTURE_2D, 0);
        glActiveTexture(GL_TEXTURE0);
    }

    void cleanup() {
        glDeleteTextures(1, &ID);
    }

    static Texture* create(unsigned int width, unsigned int height, TextureType type = TEXTURE_DIFFUSE,
                           GLenum format = GL_RGB, GLint wrap = GL_CLAMP_TO_EDGE, GLint filter = GL_LINEAR,
                           unsigned char* data = nullptr) {
        return new Texture(width, height, type, format, wrap, filter, data);
    }

    static Texture* create(const std::string path, TextureType type = TEXTURE_DIFFUSE) {
        return new Texture(path, type);
    }

protected:
    Texture(unsigned int width, unsigned int height, TextureType type = TEXTURE_DIFFUSE,
            GLenum format = GL_RGB, GLint wrap = GL_CLAMP_TO_EDGE, GLint filter = GL_LINEAR,
            unsigned char* data = nullptr)
                : width(width), height(height), type(type) {
        glGenTextures(1, &ID);
        glBindTexture(GL_TEXTURE_2D, ID);
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
    }

    Texture(const std::string path, TextureType type = TEXTURE_DIFFUSE)
            : path(path), type(type) {
        loadFromFile(path.c_str());
    }

    ~Texture() {
        cleanup();
    }

private:
    void loadFromFile(const std::string path);
};

#endif // TEXTURE_H
