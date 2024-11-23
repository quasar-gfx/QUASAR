#include <Utils/FileIO.h>
#include <Texture.h>

void Texture::loadFromData(unsigned char* data) {
    glGenTextures(1, &ID);

    glPixelStorei(GL_UNPACK_ALIGNMENT, alignment);

    glBindTexture(target, ID);
    if (!multiSampled) {
        glTexImage2D(target, 0, internalFormat, width, height, 0, format, type, data);
    }
#ifdef GL_CORE
    else {
        glTexImage2DMultisample(target, 4, internalFormat, width, height, GL_TRUE);
    }
#endif // gles does not have glTexImage2DMultisample
    glTexParameteri(target, GL_TEXTURE_WRAP_S, wrapS);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, wrapT);
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, minFilter);
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, magFilter);
    if (minFilter == GL_LINEAR_MIPMAP_LINEAR || minFilter == GL_LINEAR_MIPMAP_NEAREST) {
        glGenerateMipmap(target);
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
    if (type == GL_UNSIGNED_BYTE) {
        data = FileIO::loadImage(path, &texWidth, &texHeight, &texChannels);
    }
    else if (type == GL_FLOAT) {
        data = FileIO::loadImageHDR(path, &texWidth, &texHeight, &texChannels);
    }

    if (data) {
        this->width = texWidth;
        this->height = texHeight;

        if (texChannels == 1) {
            if (type == GL_UNSIGNED_BYTE) {
                internalFormat = GL_R8;
            }
            else {
                internalFormat = GL_R16F;
            }
            format = GL_RED;
        }
        else if (texChannels == 3) {
            if (type == GL_UNSIGNED_BYTE) {
                internalFormat = params.gammaCorrected ? GL_SRGB8 : GL_RGB;
            }
            else {
                internalFormat = GL_RGB16F;
            }
            format = GL_RGB;
        }
        else if (texChannels == 4) {
            if (type == GL_UNSIGNED_BYTE) {
                internalFormat = params.gammaCorrected ? GL_SRGB8_ALPHA8 : GL_RGBA;
            }
            else {
                internalFormat = GL_RGBA16F;
            }
            format = GL_RGBA;
        }

        loadFromData(static_cast<unsigned char*>(data));

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

void Texture::readPixels(unsigned char* data, bool readAsFloat) {
    bind(0);
    glGetTexImage(target, 0, GL_RGBA, !readAsFloat ? GL_UNSIGNED_BYTE : GL_FLOAT, data);
    unbind();
}

void Texture::saveAsPNG(const std::string &filename) {
    std::vector<unsigned char> data(width * height * 4);
    readPixels(data.data());

    FileIO::flipVerticallyOnWrite(true);
    FileIO::saveAsPNG(filename, width, height, 4, data.data());
}

void Texture::saveAsJPG(const std::string &filename, int quality) {
    std::vector<unsigned char> data(width * height * 4);
    readPixels(data.data());

    FileIO::flipVerticallyOnWrite(true);
    FileIO::saveAsJPG(filename, width, height, 4, data.data(), quality);
}

void Texture::saveAsHDR(const std::string &filename) {
    std::vector<float> data(width * height * 4);
    readPixels(reinterpret_cast<unsigned char*>(data.data()), true);

    FileIO::flipVerticallyOnWrite(true);
    FileIO::saveAsHDR(filename, width, height, 4, data.data());
}

#ifdef GL_CORE
void Texture::saveDepthToFile(const std::string &filename) {
    std::ofstream depthFile;
    depthFile.open(filename, std::ios::out | std::ios::binary);

    std::vector<float> data(width * height);

    bind(0);
    glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, data.data());
    unbind();

    if (depthFile.is_open()) {
        depthFile.write(reinterpret_cast<const char*>(data.data()), width * height * sizeof(float));
    }

    depthFile.close();
}
#endif
