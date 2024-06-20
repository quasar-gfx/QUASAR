#ifndef RENDER_TARGET_H
#define RENDER_TARGET_H

#include <RenderTargets/RenderTargetBase.h>

class RenderTarget : public RenderTargetBase {
public:
    Texture colorBuffer;
    Texture depthBuffer;

    explicit RenderTarget(const RenderTargetCreateParams &params)
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
            , depthBuffer({
                .width = width,
                .height = height,
                .internalFormat = GL_DEPTH_COMPONENT24,
                .format = GL_DEPTH_COMPONENT,
                .type = GL_FLOAT,
                .minFilter = GL_NEAREST,
                .magFilter = GL_NEAREST,
                .multiSampled = params.multiSampled
            }) {
        framebuffer.bind();
        framebuffer.attachTexture(colorBuffer, GL_COLOR_ATTACHMENT0);
        framebuffer.attachTexture(depthBuffer, GL_DEPTH_ATTACHMENT);

        if (!framebuffer.checkStatus()) {
            throw std::runtime_error("Framebuffer is not complete!");
        }

        framebuffer.unbind();
    }

    void blitToRenderTarget(RenderTarget &target) {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer.ID);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, target.framebuffer.ID);
        glBlitFramebuffer(0, 0, width, height, 0, 0, target.width, target.height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    }

    void resize(unsigned int width, unsigned int height) override {
        this->width = width;
        this->height = height;

        colorBuffer.resize(width, height);
        depthBuffer.resize(width, height);
    }
};

#endif // RENDER_TARGET_H
