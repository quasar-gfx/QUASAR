#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <Texture.h>

void Texture::create(const TextureCreateParams &params) {
    this->width = params.width;
    this->height = params.height;
    glGenTextures(1, &ID);
    glBindTexture(GL_TEXTURE_2D, ID);
    glTexImage2D(GL_TEXTURE_2D, 0, params.internalFormat, width, height, 0, params.format, params.type, params.data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, params.wrapS);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, params.wrapT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, params.minFilter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, params.magFilter);
    if (params.minFilter == GL_LINEAR_MIPMAP_LINEAR || params.minFilter == GL_LINEAR_MIPMAP_NEAREST) {
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    if (params.hasBorder) {
        float borderColor[] = { 1.0, 1.0, 1.0, 1.0 };
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    }
}

void Texture::loadFromFile(const TextureCreateParams &params) {
    stbi_set_flip_vertically_on_load(params.flipped);

    int texWidth, texHeight, texChannels;
    void* data = nullptr;
    if (params.type == GL_UNSIGNED_BYTE) {
        data = stbi_load(params.path.c_str(), &texWidth, &texHeight, &texChannels, 0);
    }
    else if (params.type == GL_FLOAT) {
        data = stbi_loadf(params.path.c_str(), &texWidth, &texHeight, &texChannels, 0);
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
            internalFormat = params.gammaCorrected ? GL_SRGB : GL_RGB;
            format = GL_RGB;
        }
        else if (texChannels == 4) {
            internalFormat = params.gammaCorrected ? GL_SRGB_ALPHA : GL_RGBA;
            format = GL_RGBA;
        }

        glBindTexture(GL_TEXTURE_2D, ID);
        glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, format, params.type, data);
        glGenerateMipmap(GL_TEXTURE_2D);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, params.wrapS);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, params.wrapT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, params.minFilter);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, params.magFilter);

        stbi_image_free(data);
    }
    else {
        throw std::runtime_error("Texture failed to load at path: " + params.path);
        stbi_image_free(data);
    }
}

void Texture::saveTextureToPNG(const char* filename) {
    unsigned char* data = new unsigned char[width * height * 4];

    bind(0);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, data);
    unbind();

    for (int y = 0; y < height / 2; y++) {
        for (int x = 0; x < width * 4; x++) {
            unsigned char temp = data[y * width * 4 + x];
            data[y * width * 4 + x] = data[(height - y - 1) * width * 4 + x];
            data[(height - y - 1) * width * 4 + x] = temp;
        }
    }

    stbi_write_png(filename, width, height, 4, data, width * 4);

    delete[] data;
}
