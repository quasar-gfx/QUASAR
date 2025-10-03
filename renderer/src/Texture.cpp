#include <Texture.h>
#include <Utils/FileIO.h>

using namespace quasar;

Texture::Texture() {
    target = GL_TEXTURE_2D;
}

Texture::Texture(const TextureDataCreateParams& params)
    : width(params.width)
    , height(params.height)
    , internalFormat(params.internalFormat)
    , format(params.format)
    , type(params.type)
    , wrapS(params.wrapS)
    , wrapT(params.wrapT)
    , minFilter(params.minFilter)
    , magFilter(params.magFilter)
    , alignment(params.alignment)
    , multiSampled(params.multiSampled)
    , numSamples(params.numSamples)
    , array(params.array)
    , arrayLayers(params.arrayLayers)
{
    glGenTextures(1, &ID);

    if (params.array) {
        target = GL_TEXTURE_2D_ARRAY;
    }
    else if (params.multiSampled) {
        target = GL_TEXTURE_2D_MULTISAMPLE;
    }
    else {
        target = GL_TEXTURE_2D;
    }

    if (format == GL_RED) {
        channels = 1;
    }
    else if (format == GL_RG) {
        channels = 2;
    }
    else if (format == GL_RGB) {
        channels = 3;
    }
    else if (format == GL_RGBA) {
        channels = 4;
    }

    loadFromData(params.data, true);

    if (params.hasBorder) {
        glTexParameterfv(target, GL_TEXTURE_BORDER_COLOR, glm::value_ptr(params.borderColor));
    }
}

Texture::Texture(const TextureFileCreateParams& params)
    : type(params.type)
    , wrapS(params.wrapS)
    , wrapT(params.wrapT)
    , minFilter(params.minFilter)
    , magFilter(params.magFilter)
    , alignment(params.alignment)
    , multiSampled(params.multiSampled)
    , numSamples(params.numSamples)
    , array(params.array)
    , arrayLayers(params.arrayLayers)
{
    glGenTextures(1, &ID);

    if (params.array) {
        target = GL_TEXTURE_2D_ARRAY;
    }
    else if (params.multiSampled) {
        target = GL_TEXTURE_2D_MULTISAMPLE;
    }
    else {
        target = GL_TEXTURE_2D;
    }

    loadFromFile(params.path, params.flipVertically, params.gammaCorrected);
}

Texture::~Texture() {
    cleanup();
}

void Texture::loadFromData(const void* data, bool resize) {
    glPixelStorei(GL_UNPACK_ALIGNMENT, alignment);
    glBindTexture(target, ID);

    if (!multiSampled) {
        glTexParameteri(target, GL_TEXTURE_WRAP_S, wrapS);
        glTexParameteri(target, GL_TEXTURE_WRAP_T, wrapT);
        glTexParameteri(target, GL_TEXTURE_MIN_FILTER, minFilter);
        glTexParameteri(target, GL_TEXTURE_MAG_FILTER, magFilter);

        if (!array) {
            if (resize || data == nullptr) {
                glTexImage2D(target, 0, internalFormat, width, height, 0, format, type, data);
            }
            else {
                glTexSubImage2D(target, 0, 0, 0, width, height, format, type, data);
            }
        }
        else {
            if (resize || data == nullptr) {
                glTexImage3D(target, 0, internalFormat, width, height, arrayLayers, 0, format, type, data);
            }
            else {
                glTexSubImage3D(target, 0, 0, 0, 0, width, height, arrayLayers, format, type, data);
            }
        }

        if (minFilter == GL_LINEAR_MIPMAP_LINEAR || minFilter == GL_LINEAR_MIPMAP_NEAREST) {
            glGenerateMipmap(target);
        }
    }
#ifdef GL_CORE
    else {
        glTexImage2DMultisample(target, numSamples, internalFormat, width, height, GL_TRUE);
    }
#endif
    glBindTexture(target, 0);
}

void Texture::loadFromFile(const std::string& path, bool flipVertically, bool gammaCorrected) {
    std::string resolvedPath = path;

    if (!resolvedPath.empty() && resolvedPath[0] == '~') {
        const char* home = getenv("HOME");
        if (home) {
            resolvedPath.replace(0, 1, home);
        }
    }

    FileIO::flipVerticallyOnLoad(flipVertically);

    int texWidth, texHeight, texChannels;
    void* data = nullptr;
    if (type == GL_UNSIGNED_BYTE) {
        data = FileIO::loadImage(resolvedPath, &texWidth, &texHeight, &texChannels);
    }
    else if (type == GL_FLOAT || type == GL_HALF_FLOAT) {
        data = FileIO::loadImageHDR(resolvedPath, &texWidth, &texHeight, &texChannels);
    }
    if (!data) {
        throw std::runtime_error("Texture failed to load at path: " + resolvedPath);
    }

    width = texWidth;
    height = texHeight;
    channels = texChannels;

    switch (texChannels) {
        case 1:
            internalFormat = (type == GL_UNSIGNED_BYTE)
                ? GL_R8
                : GL_R16F;
            format = GL_RED;
            break;
        case 2:
            internalFormat = (type == GL_UNSIGNED_BYTE)
                ? GL_RG8
                : GL_RG16F;
            format = GL_RG;
            break;
        case 3:
            internalFormat = (type == GL_UNSIGNED_BYTE)
                ? (gammaCorrected ? GL_SRGB8 : GL_RGB)
                : GL_RGB16F;
            format = GL_RGB;
            break;
        case 4:
            internalFormat = (type == GL_UNSIGNED_BYTE)
                ? (gammaCorrected ? GL_SRGB8_ALPHA8 : GL_RGBA)
                : GL_RGBA16F;
            format = GL_RGBA;
            break;
    }

    loadFromData(data, true);
    FileIO::freeImage(data);
}

void Texture::resize(uint width, uint height) {
    if (this->width == width && this->height == height) {
        return;
    }

    this->width = width;
    this->height = height;
    loadFromData(nullptr, true);
}

void Texture::readPixels(unsigned char* data, bool readAsFloat) {
    bind(0);
    glReadPixels(0, 0, width, height, format, readAsFloat ? GL_FLOAT : GL_UNSIGNED_BYTE, data);
    unbind();
}

void Texture::writeToPNG(const std::string& filename) {
    std::vector<unsigned char> data(width * height * channels);
    readPixels(data.data());

    FileIO::flipVerticallyOnWrite(true);
    FileIO::writeToPNG(filename, width, height, channels, data.data());
}

void Texture::writeToJPG(const std::string& filename, int quality) {
    std::vector<unsigned char> data(width * height * channels);
    readPixels(data.data());

    FileIO::flipVerticallyOnWrite(true);
    FileIO::writeToJPG(filename, width, height, channels, data.data(), quality);
}

void Texture::writeToHDR(const std::string& filename) {
    std::vector<float> data(width * height * channels);
    readPixels(reinterpret_cast<unsigned char*>(data.data()), true);

    FileIO::flipVerticallyOnWrite(true);
    FileIO::writeToHDR(filename, width, height, channels, data.data());
}

#ifdef GL_CORE
void Texture::saveDepthToFile(const std::string& filename) {
    std::vector<float> data(width * height);

    bind(0);
    glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, data.data());
    unbind();

    FileIO::writeToBinaryFile(filename, data.data(), data.size());
}
#endif

void Texture::writeJPGToMemory(std::vector<unsigned char>& outputData, int quality) {
    outputData.resize(width * height * channels);
    readPixels(outputData.data());

    FileIO::flipVerticallyOnWrite(true);
    FileIO::writeJPGToMemory(outputData, width, height, channels, outputData.data(), quality);
}
