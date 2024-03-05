#ifndef TEXTURE_H
#define TEXTURE_H

#include <glad/glad.h>
#include <string>

#include <OpenGLObject.h>

enum TextureType {
    TEXTURE_DIFFUSE  = 0,
    TEXTURE_SPECULAR = 1,
    TEXTURE_NORMAL   = 2,
    TEXTURE_HEIGHT   = 3
};

class Texture : public OpenGLObject {
public:
    unsigned int width, height;

    std::string path;

    void bind() {
        bind(0);
    }

    void bind(unsigned int slot = 0) {
        glActiveTexture(GL_TEXTURE0 + slot);
        glBindTexture(GL_TEXTURE_2D, ID);
    }

    void unbind() {
        unbind(0);
    }

    void unbind(unsigned int slot) {
        glActiveTexture(GL_TEXTURE0 + slot);
        glBindTexture(GL_TEXTURE_2D, 0);
        glActiveTexture(GL_TEXTURE0);
    }

    void cleanup() {
        glDeleteTextures(1, &ID);
    }

    static Texture* create(unsigned int width, unsigned int height,
            GLint internalFormat = GL_RGB,
            GLenum format = GL_RGB,
            GLenum type = GL_UNSIGNED_BYTE,
            GLint wrapS = GL_CLAMP_TO_EDGE, GLint wrapT = GL_CLAMP_TO_EDGE,
            GLint minFilter = GL_LINEAR, GLint magFilter = GL_LINEAR,
            unsigned char* data = nullptr) {
        return new Texture(width, height, internalFormat, format, type, wrapS, wrapT, minFilter, magFilter, data);
    }

    static Texture* create(const std::string path,
            GLint internalFormat = GL_RGB,
            /* format is inferred from image loaded by stb */
            GLenum type = GL_UNSIGNED_BYTE,
            GLint wrapS = GL_REPEAT, GLint wrapT = GL_REPEAT,
            GLint minFilter = GL_LINEAR_MIPMAP_LINEAR, GLint magFilter = GL_LINEAR) {
        return new Texture(path, internalFormat, type, wrapS, wrapT, minFilter, magFilter);
    }

protected:
    Texture(unsigned int width, unsigned int height,
            GLint internalFormat = GL_RGB,
            GLenum format = GL_RGB,
            GLenum type = GL_UNSIGNED_BYTE,
            GLint wrapS = GL_CLAMP_TO_EDGE, GLint wrapT = GL_CLAMP_TO_EDGE,
            GLint minFilter = GL_LINEAR, GLint magFilter = GL_LINEAR,
            unsigned char* data = nullptr)
                : width(width), height(height) {
        glGenTextures(1, &ID);
        glBindTexture(GL_TEXTURE_2D, ID);
        glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, format, type, data);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapS);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magFilter);
    }

    Texture(const std::string path,
            GLint internalFormat = GL_RGB,
            GLenum type = GL_UNSIGNED_BYTE,
            GLint wrapS = GL_REPEAT, GLint wrapT = GL_REPEAT,
            GLint minFilter = GL_LINEAR_MIPMAP_LINEAR, GLint magFilter = GL_LINEAR)
                : path(path) {
        loadFromFile(path.c_str(), internalFormat, type, wrapS, wrapT, minFilter, magFilter);
    }

    ~Texture() {
        cleanup();
    }

private:
    void loadFromFile(const std::string path,
        GLint internalFormat = GL_RGB, GLenum type = GL_UNSIGNED_BYTE,
        GLint wrapS = GL_REPEAT, GLint wrapT = GL_REPEAT,
        GLint minFilter = GL_LINEAR_MIPMAP_LINEAR, GLint magFilter = GL_LINEAR);
};

#endif // TEXTURE_H
