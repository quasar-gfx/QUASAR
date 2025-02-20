#ifndef RENDER_TARGET_H
#define RENDER_TARGET_H

#include <RenderTargets/RenderTargetBase.h>

class RenderTarget : public RenderTargetBase {
public:
    Texture colorBuffer;
    Texture depthStencilBuffer;

    RenderTarget(const RenderTargetCreateParams &params)
            : RenderTargetBase(params)
            , colorBuffer({
                .width = width,
                .height = height,
                .internalFormat = params.internalFormat,
                .format = params.format,
                .type = params.type,
                .wrapS = params.wrapS,
                .wrapT = params.wrapT,
                .minFilter = params.minFilter,
                .magFilter = params.magFilter,
                .multiSampled = params.multiSampled
            })
            , depthStencilBuffer({
                .width = width,
                .height = height,
                .internalFormat = GL_DEPTH24_STENCIL8,
                .format = GL_DEPTH_STENCIL,
                .type = GL_UNSIGNED_INT_24_8,
                .wrapS = GL_CLAMP_TO_EDGE,
                .wrapT = GL_CLAMP_TO_EDGE,
                .minFilter = GL_NEAREST,
                .magFilter = GL_NEAREST,
                .multiSampled = params.multiSampled
            }) {
        framebuffer.bind();
        framebuffer.attachTexture(colorBuffer, GL_COLOR_ATTACHMENT0);
        framebuffer.attachTexture(depthStencilBuffer, GL_DEPTH_STENCIL_ATTACHMENT);

        if (!framebuffer.checkStatus()) {
            throw std::runtime_error("Framebuffer is not complete!");
        }

        framebuffer.unbind();
    }

#ifdef GL_CORE
    void blitToRenderTarget(RenderTargetBase &rt) {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer.ID);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, rt.getFramebufferID());

        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glDrawBuffer(GL_COLOR_ATTACHMENT0);
        glBlitFramebuffer(0, 0, width, height, 0, 0, rt.width, rt.height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    }
#endif

    void blitToRenderTarget(RenderTarget &rt) {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer.ID);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, rt.framebuffer.ID);
        glBlitFramebuffer(0, 0, width, height, 0, 0, rt.width, rt.height, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT, GL_NEAREST);
    }

    void resize(unsigned int width, unsigned int height) override {
        this->width = width;
        this->height = height;

        colorBuffer.resize(width, height);
        depthStencilBuffer.resize(width, height);
    }

    void readPixels(unsigned char* data, bool readAsFloat = false) {
        bind();
        colorBuffer.readPixels(data, readAsFloat);
        unbind();
    }

    void saveColorAsPNG(const std::string &path) {
        bind();
        colorBuffer.saveAsPNG(path);
        unbind();
    }

    void saveColorAsJPG(const std::string &path, int quality = 90) {
        bind();
        colorBuffer.saveAsJPG(path, quality);
        unbind();
    }

    void saveColorAsHDR(const std::string &path) {
        bind();
        colorBuffer.saveAsHDR(path);
        unbind();
    }
};

#endif // RENDER_TARGET_H
