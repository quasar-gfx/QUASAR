#ifndef RENDER_TARGET_H
#define RENDER_TARGET_H

#include <RenderTargets/RenderTargetBase.h>

namespace quasar {

class RenderTarget : public RenderTargetBase {
public:
    Texture colorTexture;
    Texture depthStencilTexture;

    RenderTarget(const RenderTargetCreateParams& params)
        : RenderTargetBase(params)
        , colorTexture({
            .width = width,
            .height = height,
            .internalFormat = params.internalFormat,
            .format = params.format,
            .type = params.type,
            .wrapS = params.wrapS,
            .wrapT = params.wrapT,
            .minFilter = params.minFilter,
            .magFilter = params.magFilter,
            .multiSampled = params.multiSampled,
        })
        , depthStencilTexture({
            .width = width,
            .height = height,
            .internalFormat = GL_DEPTH32F_STENCIL8,
            .format = GL_DEPTH_STENCIL,
            .type = GL_FLOAT_32_UNSIGNED_INT_24_8_REV,
            .wrapS = GL_CLAMP_TO_EDGE,
            .wrapT = GL_CLAMP_TO_EDGE,
            .minFilter = GL_NEAREST,
            .magFilter = GL_NEAREST,
            .multiSampled = params.multiSampled,
            .numSamples = params.numSamples,
        })
    {
        framebuffer.bind();
        framebuffer.attachTexture(colorTexture, GL_COLOR_ATTACHMENT0);
        framebuffer.attachTexture(depthStencilTexture, GL_DEPTH_STENCIL_ATTACHMENT);

        if (!framebuffer.checkStatus()) {
            throw std::runtime_error("Framebuffer is not complete!");
        }

        framebuffer.unbind();
    }
    ~RenderTarget() = default;

    void blit(RenderTargetBase& rt) {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer.ID);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, rt.getFramebufferID());

        glReadBuffer(GL_COLOR_ATTACHMENT0);
        GLenum drawBuffers[] = { GL_COLOR_ATTACHMENT0 };
        glDrawBuffers(1, drawBuffers);
        glBlitFramebuffer(0, 0, width, height, 0, 0, rt.width, rt.height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    }

    void blit(RenderTarget& rt) {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer.ID);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, rt.framebuffer.ID);
        glBlitFramebuffer(0, 0, width, height, 0, 0, rt.width, rt.height, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT, GL_NEAREST);
    }

    void clear(uint32_t clearMask) override {
        if (clearMask & GL_COLOR_BUFFER_BIT) {
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        }
        if (clearMask & (GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)) {
#ifdef GL_CORE
            glClearDepth(1.0f);
#else
            glClearDepthf(1.0f);
#endif
            glClearStencil(0);
        }
        glClear(clearMask);
    }

    void resize(uint width, uint height) override {
        this->width = width;
        this->height = height;

        colorTexture.resize(width, height);
        depthStencilTexture.resize(width, height);
    }

    void readPixels(unsigned char* data, bool readAsFloat = false) {
        bind();
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        colorTexture.readPixels(data, readAsFloat);
        unbind();
    }

    void writeColorAsPNG(const std::string& path) {
        bind();
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        colorTexture.writeToPNG(path);
        unbind();
    }

    void writeColorAsJPG(const std::string& path, int quality = 85) {
        bind();
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        colorTexture.writeToJPG(path, quality);
        unbind();
    }

    void writeColorAsHDR(const std::string& path) {
        bind();
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        colorTexture.writeToHDR(path);
        unbind();
    }

    void writeColorAsEXR(const std::string& path) {
        bind();
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        colorTexture.writeToEXR(path);
        unbind();
    }

    void writeColorJPGToMemory(std::vector<unsigned char>& outputData, int quality = 85) {
        bind();
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        colorTexture.writeJPGToMemory(outputData, quality);
        unbind();
    }
};

} // namespace quasar

#endif // RENDER_TARGET_H
