#include <Utils/FileIO.h>
#include <Texture.h>

void Texture::loadFromData(const TextureDataCreateParams &params) {
    glGenTextures(1, &ID);

    glBindTexture(target, ID);
    if (!multiSampled) {
        glTexImage2D(target, 0, params.internalFormat, width, height, 0, params.format, params.type, params.data);
    }
#ifdef GL_CORE
    else {
        glTexImage2DMultisample(target, 4, params.internalFormat, width, height, GL_TRUE);
    }
#endif // gles does not have glTexImage2DMultisample
    glTexParameteri(target, GL_TEXTURE_WRAP_S, params.wrapS);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, params.wrapT);
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, params.minFilter);
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, params.magFilter);
    if (params.minFilter == GL_LINEAR_MIPMAP_LINEAR || params.minFilter == GL_LINEAR_MIPMAP_NEAREST) {
        glGenerateMipmap(target);
    }
    if (params.hasBorder) {
        float borderColor[] = { 1.0, 1.0, 1.0, 1.0 };
        glTexParameterfv(target, GL_TEXTURE_BORDER_COLOR, borderColor);
    }
}

void Texture::loadFromFile(const TextureFileCreateParams &params) {
    std::string path = params.path;

    // use absolute path if path starts with ~/
    if (path[0] == '~') {
        char* home = getenv("HOME");
        if (home != nullptr) {
            path.replace(0, 1, home);
        }
    }

    FileIO::flipVerticallyOnLoad(params.flipVertically);

    int texWidth, texHeight, texChannels;
    void* data = nullptr;
    if (params.type == GL_UNSIGNED_BYTE) {
        data = FileIO::loadImage(path, &texWidth, &texHeight, &texChannels);
    }
    else if (params.type == GL_FLOAT) {
        data = FileIO::loadImageHDR(path, &texWidth, &texHeight, &texChannels);
    }

    if (data) {
        glGenTextures(1, &ID);

        this->width = texWidth;
        this->height = texHeight;

        GLenum internalFormat, format;
        if (texChannels == 1) {
#ifdef GL_CORE
            internalFormat = GL_RED;
            format = GL_RED;
#else
            internalFormat = GL_LUMINANCE;
            format = GL_LUMINANCE;
#endif
        }
        else if (texChannels == 3) {
            internalFormat = params.gammaCorrected ? GL_SRGB : GL_RGB;
            format = GL_RGB;
        }
        else if (texChannels == 4) {
#ifdef GL_CORE
            internalFormat = params.gammaCorrected ? GL_SRGB_ALPHA : GL_RGBA;
#else
            internalFormat = GL_RGBA;
#endif
            format = GL_RGBA;
        }

        glBindTexture(target, ID);
        if (!multiSampled) {
            glTexImage2D(target, 0, internalFormat, width, height, 0, format, params.type, data);
        }
#ifdef GL_CORE
        else {
            glTexImage2DMultisample(target, 4, internalFormat, width, height, GL_TRUE);
        }
#endif

        glTexParameteri(target, GL_TEXTURE_WRAP_S, params.wrapS);
        glTexParameteri(target, GL_TEXTURE_WRAP_T, params.wrapT);
        glTexParameteri(target, GL_TEXTURE_MIN_FILTER, params.minFilter);
        glTexParameteri(target, GL_TEXTURE_MAG_FILTER, params.magFilter);
        if (params.minFilter == GL_LINEAR_MIPMAP_LINEAR || params.minFilter == GL_LINEAR_MIPMAP_NEAREST) {
            glGenerateMipmap(target);
        }

        FileIO::freeImage(data);
    }
    else {
        throw std::runtime_error("Texture failed to load at path: " + params.path);
    }
}

void Texture::resize(unsigned int width, unsigned int height) {
    this->width = width;
    this->height = height;

    bind(0);
    if (!multiSampled) {
        glTexImage2D(target, 0, internalFormat, width, height, 0, format, type, nullptr);
    }
#ifdef GL_CORE
    else {
        glTexImage2DMultisample(target, 4, internalFormat, width, height, GL_TRUE);
    }
#endif
    if (minFilter == GL_LINEAR_MIPMAP_LINEAR || minFilter == GL_LINEAR_MIPMAP_NEAREST) {
        glGenerateMipmap(target);
    }
    unbind();
}

void Texture::saveAsPNG(const std::string &filename) {

    unsigned char* data = new unsigned char[width * height * 4];

    bind(0);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, data);
    unbind();

    FileIO::flipVerticallyOnWrite(true);
    FileIO::saveAsPNG(filename, width, height, 4, data);

    delete[] data;
}

void Texture::saveAsJPG(const std::string &filename, int quality) {

    unsigned char* data = new unsigned char[width * height * 4];

    bind(0);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, data);
    unbind();

    FileIO::flipVerticallyOnWrite(true);
    FileIO::saveAsJPG(filename, width, height, 4, data, quality);

    delete[] data;
}

void Texture::saveAsHDR(const std::string &filename) {

    float* data = new float[width * height * 4];

    bind(0);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_FLOAT, data);
    unbind();

    FileIO::flipVerticallyOnWrite(true);
    FileIO::saveAsHDR(filename, width, height, 4, data);

    delete[] data;
}

#ifdef GL_CORE
void Texture::saveDepthToFile(const std::string &filename) {
    std::ofstream depthFile;
    depthFile.open(filename, std::ios::out | std::ios::binary);

    GLuint pbo;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_PACK_BUFFER, width * height * sizeof(float), NULL, GL_STREAM_READ);

    bind(0);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(0, 0, width, height, GL_RED, GL_FLOAT, 0);

    float* data = (float*)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);

    if (depthFile.is_open()) {
        depthFile.write(reinterpret_cast<const char*>(data), width * height * sizeof(float));
    }

    glUnmapBuffer(GL_PIXEL_PACK_BUFFER);

    depthFile.close();
}
#endif
