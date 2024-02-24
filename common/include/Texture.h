#ifndef TEXTURE_H
#define TEXTURE_H

#include "glad/glad.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <string>

#include "OpenGLObject.h"

class Texture : public OpenGLObject {
public:
    GLuint ID;

    unsigned int width, height;

    Texture(unsigned int width, unsigned int height,
            GLenum format = GL_RGB, GLint wrap = GL_CLAMP_TO_EDGE, GLint filter = GL_LINEAR,
            unsigned char* data = nullptr) : width(width), height(height) {
        glGenTextures(1, &ID);
        glBindTexture(GL_TEXTURE_2D, ID);
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
    }

    Texture(const std::string path) {
        loadFromFile(path.c_str());
    }

    ~Texture() {
        cleanup();
    }

    void bind() {
        bind(0);
    }

    void bind(unsigned int slot) {
        glActiveTexture(GL_TEXTURE0 + slot);
        glBindTexture(GL_TEXTURE_2D, ID);
    }

    void unbind() {
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    void cleanup() {
        glDeleteTextures(1, &ID);
    }

private:
    void loadFromFile(char const * path) {
        glGenTextures(1, &ID);

        int width, height, nrComponents;
        unsigned char* data = stbi_load(path, &width, &height, &nrComponents, 0);
        if (data) {
            GLenum format;
            if (nrComponents == 1)
                format = GL_RED;
            else if (nrComponents == 3)
                format = GL_RGB;
            else if (nrComponents == 4)
                format = GL_RGBA;

            glBindTexture(GL_TEXTURE_2D, ID);
            glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
            glGenerateMipmap(GL_TEXTURE_2D);

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

            stbi_image_free(data);
        }
        else {
            std::cerr << "Texture failed to load at path: " << path << std::endl;
            stbi_image_free(data);
        }
    }
};

#endif // TEXTURE_H
