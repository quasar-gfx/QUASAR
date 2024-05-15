#ifndef RENDER_TARGET_H
#define RENDER_TARGET_H

#include <RenderTargets/RenderTargetBase.h>

class RenderTarget : public RenderTargetBase {
public:
    Texture colorBuffer;
    Texture depthBuffer;

    explicit RenderTarget() = default;

    explicit RenderTarget(const RenderTargetCreateParams &params) {
        init(params);
    }

    void init(const RenderTargetCreateParams &params) override {
        width = params.width;
        height = params.height;

        colorBuffer = Texture({
            .width = width,
            .height = height,
            .internalFormat = params.internalFormat,
            .format = params.format,
            .type = params.type,
            .wrapS = params.wrapS,
            .wrapT = params.wrapT,
            .minFilter = params.minFilter,
            .magFilter = params.magFilter
        });

        depthBuffer = Texture({
            .width = width,
            .height = height,
            .internalFormat = GL_DEPTH_COMPONENT32F,
            .format = GL_DEPTH_COMPONENT,
            .type = GL_FLOAT,
            .minFilter = GL_NEAREST,
            .magFilter = GL_NEAREST
        });

        framebuffer.init();

        framebuffer.bind();
        framebuffer.attachTexture(colorBuffer, GL_COLOR_ATTACHMENT0);
        framebuffer.attachTexture(depthBuffer, GL_DEPTH_ATTACHMENT);

        if (!framebuffer.checkStatus()) {
            throw std::runtime_error("Framebuffer is not complete!");
        }

        framebuffer.unbind();
    }

    void resize(unsigned int width, unsigned int height) override {
        this->width = width;
        this->height = height;

        colorBuffer.resize(width, height);
        depthBuffer.resize(width, height);
    }
};

#endif // RENDER_TARGET_H
