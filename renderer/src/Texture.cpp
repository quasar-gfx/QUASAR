#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <Texture.h>

Texture::Texture(unsigned int width, unsigned int height,
        GLint internalFormat, GLenum format,
        GLenum type,
        GLint wrapS, GLint wrapT,
        GLint minFilter, GLint magFilter,
        unsigned char* data) {
    this->width = width;
    this->height = height;
    glGenTextures(1, &ID);
    glBindTexture(GL_TEXTURE_2D, ID);
    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, format, type, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapS);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magFilter);
}

Texture::Texture(const std::string &path, GLenum type, GLint wrapS, GLint wrapT, GLint minFilter, GLint magFilter, bool flipTexture) {
    this->path = path;
    this->flipped = flipTexture;

    stbi_set_flip_vertically_on_load(flipTexture);

    int texWidth, texHeight, texChannels;
    void* data = nullptr;
    if (type == GL_UNSIGNED_BYTE) {
        data = stbi_load(path.c_str(), &texWidth, &texHeight, &texChannels, 0);
    }
    else if (type == GL_FLOAT) {
        data = stbi_loadf(path.c_str(), &texWidth, &texHeight, &texChannels, 0);
    }

    if (data) {
        glGenTextures(1, &ID);

        this->width = texWidth;
        this->height = texHeight;

        GLenum internalFormat, format;
        if (texChannels == 1) {
            internalFormat = GL_RED;
            format = GL_RED;
        }
        else if (texChannels == 3) {
            internalFormat = GL_RGB;
            format = GL_RGB;
        }
        else if (texChannels == 4) {
            internalFormat = GL_RGBA;
            format = GL_RGBA;
        }

        glBindTexture(GL_TEXTURE_2D, ID);
        glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, format, type, data);
        glGenerateMipmap(GL_TEXTURE_2D);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapS);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magFilter);

        stbi_image_free(data);
    }
    else {
        throw std::runtime_error("Texture failed to load at path: " + path);
        stbi_image_free(data);
    }
}
