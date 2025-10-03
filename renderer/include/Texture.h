#ifndef TEXTURE_H
#define TEXTURE_H

#include <string>
#include <vector>

#include <glm/gtc/type_ptr.hpp>

#include <OpenGLObject.h>

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
    bool flipVertically = false;
    bool gammaCorrected = false;
    GLint alignment = 1;
    bool multiSampled = false;
    uint numSamples = 4;
    bool array = false;
    uint arrayLayers = 2;
    std::string path = "";
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

    void loadFromFile(const std::string& path, bool flipVertically, bool gammaCorrected);
    void loadFromData(const void* data, bool resize = false);

    void cleanup() {
        glDeleteTextures(1, &ID);
    }

    void readPixels(unsigned char* data, bool readAsFloat = false);
    void writeToPNG(const std::string& filename);
    void writeToJPG(const std::string& filename, int quality = 85);
    void writeToHDR(const std::string& filename);
#ifdef GL_CORE
    void saveDepthToFile(const std::string& filename);
#endif
    void writeJPGToMemory(std::vector<unsigned char>& outputData, int quality = 85);

protected:
    GLenum target;
};

} // namespace quasar

#endif // TEXTURE_H
