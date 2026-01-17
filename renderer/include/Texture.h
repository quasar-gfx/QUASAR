#ifndef TEXTURE_H
#define TEXTURE_H

#include <string>
#include <vector>

#include <glm/gtc/type_ptr.hpp>

#include <OpenGLObject.h>
#include <Path.h>

namespace quasar {

struct TextureDataCreateParams {
    uint width = 0;
    uint height = 0;
    GLint internalFormat = GL_RGB;
    GLenum format = GL_RGB;
    GLenum type = GL_UNSIGNED_BYTE;
    GLint wrapS = GL_CLAMP_TO_EDGE;
    GLint wrapT = GL_CLAMP_TO_EDGE;
    GLint minFilter = GL_LINEAR;
    GLint magFilter = GL_LINEAR;
    bool hasBorder = false;
    glm::vec4 borderColor = glm::vec4(1.0f);
    bool gammaCorrected = false;
    GLint alignment = 4;
    bool multiSampled = false;
    uint numSamples = 4;
    bool array = false;
    uint arrayLayers = 2;
    const unsigned char* data = nullptr;
};

struct TextureFileCreateParams {
    GLenum type = GL_UNSIGNED_BYTE;
    GLint wrapS = GL_CLAMP_TO_EDGE;
    GLint wrapT = GL_CLAMP_TO_EDGE;
    GLint minFilter = GL_LINEAR;
    GLint magFilter = GL_LINEAR;
    bool flipTextureY = true;
    bool gammaCorrected = false;
    GLint alignment = 1;
    bool multiSampled = false;
    uint numSamples = 4;
    bool array = false;
    uint arrayLayers = 2;
    Path path;
};

class Texture : public OpenGLObject {
public:
    uint width, height, channels;

    GLint internalFormat;
    GLenum format;
    GLenum type;

    GLint wrapS;
    GLint wrapT;
    GLint minFilter;
    GLint magFilter;

    GLint alignment;

    bool multiSampled;
    uint numSamples;

    bool array;
    uint arrayLayers;

    Texture();
    Texture(const TextureDataCreateParams& params);
    Texture(const TextureFileCreateParams& params);
    ~Texture();

    virtual void bind() const override {
        bind(0);
    }

    virtual void bind(uint slot) const {
        glActiveTexture(GL_TEXTURE0 + slot);
        glBindTexture(target, ID);
    }

    virtual void unbind() const override {
        unbind(0);
    }

    virtual void unbind(uint slot) const {
        glActiveTexture(GL_TEXTURE0 + slot);
        glBindTexture(target, 0);
        glActiveTexture(GL_TEXTURE0);
    }

    void resize(uint width, uint height);

    void loadFromFile(const Path& path, bool flipTextureY, bool gammaCorrected);
    void loadFromData(const void* data, bool resize = false);

    void cleanup() {
        glDeleteTextures(1, &ID);
    }

    void readPixels(unsigned char* data, bool readAsFloat = false);

    void writeJPGToMemory(std::vector<unsigned char>& outputData, int quality = 85);
    void writeToPNG(const Path& filename);
    void writeToJPG(const Path& filename, int quality = 85);
    void writeToHDR(const Path& filename);
    void writeToEXR(const Path& filename, bool convertToLinear = true);
#ifdef GL_CORE
    void writeDepthToFile(const Path& filename);
#endif

protected:
    GLenum target;
};

} // namespace quasar

#endif // TEXTURE_H
