#ifndef TEXTURE_H
#define TEXTURE_H

#include <glad/glad.h>
#include <string>

#include <OpenGLObject.h>

typedef GLuint TextureID;

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

    bool flipped = false;

    Texture() = default;

    Texture(unsigned int width, unsigned int height,
            GLint internalFormat = GL_RGB,
            GLenum format = GL_RGB,
            GLenum type = GL_UNSIGNED_BYTE,
            GLint wrapS = GL_CLAMP_TO_EDGE, GLint wrapT = GL_CLAMP_TO_EDGE,
            GLint minFilter = GL_LINEAR, GLint magFilter = GL_LINEAR,
            unsigned char* data = nullptr);

    Texture(const std::string &path,
            GLenum type = GL_UNSIGNED_BYTE,
            GLint wrapS = GL_REPEAT, GLint wrapT = GL_REPEAT,
            GLint minFilter = GL_LINEAR_MIPMAP_LINEAR, GLint magFilter = GL_LINEAR,
            bool flipTexture = false);

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
};

#endif // TEXTURE_H
