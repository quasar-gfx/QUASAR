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
    bool flipped = false;
    bool hasBorder = false;
    bool gammaCorrected = false;
    unsigned char* data = nullptr;
    std::string path = "";
};

class Texture : public OpenGLObject {
public:
    unsigned int width, height;

    explicit Texture() = default;

    Texture(const TextureCreateParams &params) {
        if (params.path == "") {
            create(params);
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

    void cleanup() {
        glDeleteTextures(1, &ID);
    }

    void saveTextureToPNG(std::string filename);
    void saveDepthToFile(std::string filename);

private:
    void create(const TextureCreateParams &params);
    void loadFromFile(const TextureCreateParams &params);
};

#endif // TEXTURE_H
