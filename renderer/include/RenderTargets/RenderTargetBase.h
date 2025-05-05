#ifndef RENDER_TARGET_BASE_H
#define RENDER_TARGET_BASE_H

#include <Texture.h>
#include <CubeMap.h>
#include <Framebuffer.h>

namespace quasar {

struct RenderTargetCreateParams {
    uint width = 0;
    uint height = 0;
    GLint internalFormat = GL_RGB;
    GLenum format = GL_RGB;
    GLenum type = GL_UNSIGNED_BYTE;
    GLint wrapS = GL_CLAMP_TO_EDGE;
    GLint wrapT = GL_CLAMP_TO_EDGE;
    GLint minFilter = GL_LINEAR;
    GLint magFilter = GL_LINEAR;
    bool multiSampled = false;
    uint numSamples = 4;
};

class RenderTargetBase {
public:
    uint width, height;

    struct Scissor {
        uint x, y, width, height;

        void apply() const {
            glScissor(x, y, width, height);
        }
    } scissor;

    struct Viewport {
        uint x, y, width, height;

        void apply() const {
            glViewport(x, y, width, height);
        }
    } viewport;

    RenderTargetBase(const RenderTargetCreateParams &params) : framebuffer() {
        width = params.width;
        height = params.height;

        setViewport(0, 0, width, height);
        setScissor(0, 0, width, height);
    }
    ~RenderTargetBase() = default;

    virtual void resize(uint width, uint height) {
        this->width = width;
        this->height = height;

        setViewport(0, 0, width, height);
        setScissor(0, 0, width, height);
    }

    void blitToScreen(uint width, uint height) {
        framebuffer.blitToScreen(width, height);
    }

    void setScissor(uint x, uint y, uint width, uint height) {
        scissor = { x, y, width, height };
        scissor.apply();
    }

    void setViewport(uint x, uint y, uint width, uint height) {
        viewport = { x, y, width, height };
        viewport.apply();
    }

    GLuint getFramebufferID() const {
        return framebuffer.ID;
    }

    void bind() const {
        framebuffer.bind();
        scissor.apply();
        viewport.apply();
    }

    void unbind() const {
        framebuffer.unbind();
    }

protected:
    Framebuffer framebuffer;
};

} // namespace quasar

#endif // RENDER_TARGET_BASE_H
