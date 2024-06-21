#ifndef RENDER_BASE_TARGET_H
#define RENDER_BASE_TARGET_H

#include <Texture.h>
#include <CubeMap.h>
#include <Framebuffer.h>

struct RenderTargetCreateParams {
    unsigned int width = 0;
    unsigned int height = 0;
    GLint internalFormat = GL_RGB;
    GLenum format = GL_RGB;
    GLenum type = GL_UNSIGNED_BYTE;
    GLint wrapS = GL_CLAMP_TO_EDGE;
    GLint wrapT = GL_CLAMP_TO_EDGE;
    GLint minFilter = GL_LINEAR;
    GLint magFilter = GL_LINEAR;
    bool multiSampled = false;
};

class RenderTargetBase {
public:
    unsigned int width, height;

    explicit RenderTargetBase(const RenderTargetCreateParams &params) : framebuffer() {
        width = params.width;
        height = params.height;
    }

    ~RenderTargetBase() {
        cleanup();
    }

    virtual void resize(unsigned int width, unsigned int height) {
        this->width = width;
        this->height = height;
    }

    void blitToScreen(unsigned int width, unsigned int height) {
        framebuffer.blitToScreen(width, height);
    }

    void bind() {
        framebuffer.bind();
        glViewport(0, 0, width, height);
    }

    void unbind() {
        framebuffer.unbind();
    }

    virtual void cleanup() {
        framebuffer.cleanup();
    }

protected:
    Framebuffer framebuffer;
};

#endif // RENDER_BASE_TARGET_H
