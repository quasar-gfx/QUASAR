#ifndef TEXTURE_H
#define TEXTURE_H

#include <glad/glad.h>
#include <string>

#include <OpenGLObject.h>

typedef GLuint TextureID;

struct TextureCreateParams {
    unsigned int width = 0;
    unsigned int height = 0;
    GLint internalFormat = GL_RGB;
    GLenum format = GL_RGB;
    GLenum type = GL_UNSIGNED_BYTE;
    GLint wrapS = GL_CLAMP_TO_EDGE;
    GLint wrapT = GL_CLAMP_TO_EDGE;
    GLint minFilter = GL_LINEAR;
    GLint magFilter = GL_LINEAR;
    bool flipVertically = false;
    bool hasBorder = false;
    bool gammaCorrected = false;
    unsigned char* data = nullptr;
    std::string path = "";
};

class Texture : public OpenGLObject {
public:
    unsigned int width, height;

    GLint internalFormat = GL_RGB;
    GLenum format = GL_RGB;
    GLenum type = GL_UNSIGNED_BYTE;

    GLint wrapS = GL_CLAMP_TO_EDGE;
    GLint wrapT = GL_CLAMP_TO_EDGE;

    GLint minFilter = GL_LINEAR;
    GLint magFilter = GL_LINEAR;

    explicit Texture() = default;

    explicit Texture(const TextureCreateParams &params)
            : width(params.width), height(params.height),
              internalFormat(params.internalFormat), format(params.format),
              type(params.type),
              wrapS(params.wrapS), wrapT(params.wrapT),
              minFilter(params.minFilter), magFilter(params.magFilter) {
        if (params.path == "") {
            init(params);
        }
        else {
            loadFromFile(params);
        }
    }

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

    void resize(unsigned int width, unsigned int height) {
        this->width = width;
        this->height = height;

        glBindTexture(GL_TEXTURE_2D, ID);
        glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, format, type, nullptr);
        if (minFilter == GL_LINEAR_MIPMAP_LINEAR || minFilter == GL_LINEAR_MIPMAP_NEAREST) {
            glGenerateMipmap(GL_TEXTURE_2D);
        }
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    void cleanup() {
        glDeleteTextures(1, &ID);
    }

    void saveTextureToPNG(std::string filename);
    void saveTextureToJPG(std::string filename, int quality = 100);
    void saveTextureToHDR(std::string filename);
    void saveDepthToFile(std::string filename);

private:
    void init(const TextureCreateParams &params);
    void loadFromFile(const TextureCreateParams &params);
};

#endif // TEXTURE_H
