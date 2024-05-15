#ifndef DIR_LIGHT_SHADOW_BUFFER_H
#define DIR_LIGHT_SHADOW_BUFFER_H

#include <RenderTargets/RenderTargetBase.h>

class DirLightShadowRT : public RenderTargetBase {
public:
    Texture colorBuffer;
    Texture depthBuffer;

    explicit DirLightShadowRT() = default;

    void init(const RenderTargetCreateParams &params) override {
        width = params.width;
        height = params.height;

        depthBuffer = Texture({
            .width = width,
            .height = height,
            .internalFormat = GL_DEPTH_COMPONENT32F,
            .format = GL_DEPTH_COMPONENT,
            .type = GL_FLOAT,
            .wrapS = GL_CLAMP_TO_BORDER,
            .wrapT = GL_CLAMP_TO_BORDER,
            .minFilter = GL_LINEAR,
            .magFilter = GL_LINEAR,
            .hasBorder = true
        });
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LESS);

        framebuffer.init();
        framebuffer.bind();
        framebuffer.attachTexture(depthBuffer, GL_DEPTH_ATTACHMENT);

        glDrawBuffer(GL_NONE);
        glReadBuffer(GL_NONE);

        if (!framebuffer.checkStatus()) {
            throw std::runtime_error("DirLightShadowRT Framebuffer is not complete!");
        }

        glClear(GL_DEPTH_BUFFER_BIT);

        framebuffer.unbind();
    }
};

#endif // DIR_LIGHT_SHADOW_BUFFER_H
