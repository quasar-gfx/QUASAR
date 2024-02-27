#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <Texture.h>

void Texture::loadFromFile(const std::string path, GLint wrapS, GLint wrapT, GLint minFilter, GLint magFilter) {
    glGenTextures(1, &ID);

    int texWidth, texHeight, texChannels;
    unsigned char* data = stbi_load(path.c_str(), &texWidth, &texHeight, &texChannels, 0);
    if (data) {
        this->width = texWidth;
        this->height = texHeight;

        GLenum format;
        if (texChannels == 1)
            format = GL_RED;
        else if (texChannels == 3)
            format = GL_RGB;
        else if (texChannels == 4)
            format = GL_RGBA;

        glBindTexture(GL_TEXTURE_2D, ID);
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
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
