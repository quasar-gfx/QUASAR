#ifndef FRAME_RENDER_TARGET_H
#define FRAME_RENDER_TARGET_H

#include <RenderTargets/RenderTargetBase.h>
#include <RenderTargets/RenderTarget.h>

namespace quasar {

class FrameRenderTarget : public RenderTargetBase {
public:
    Texture colorBuffer;
    Texture normalsBuffer;
    Texture idBuffer;
    Texture depthStencilBuffer;

    FrameRenderTarget(const RenderTargetCreateParams& params)
            : RenderTargetBase(params)
            , colorBuffer({
                .width = width,
                .height = height,
                .internalFormat = GL_RGBA16F,
                .format = GL_RGBA,
                .type = GL_HALF_FLOAT,
                .wrapS = GL_CLAMP_TO_EDGE,
                .wrapT = GL_CLAMP_TO_EDGE,
                .minFilter = params.minFilter,
                .magFilter = params.magFilter,
                .multiSampled = params.multiSampled,
                .numSamples = params.numSamples
            })
            , normalsBuffer({
                .width = width,
                .height = height,
                .internalFormat = GL_RGB16F,
                .format = GL_RGB,
                .type = GL_HALF_FLOAT,
                .wrapS = GL_CLAMP_TO_EDGE,
                .wrapT = GL_CLAMP_TO_EDGE,
                .minFilter = GL_NEAREST,
                .magFilter = GL_NEAREST,
                .multiSampled = params.multiSampled,
                .numSamples = params.numSamples
            })
            , idBuffer({
                .width = width,
                .height = height,
                .internalFormat = GL_RGB32UI,
                .format = GL_RGB_INTEGER,
                .type = GL_UNSIGNED_INT,
                .wrapS = GL_CLAMP_TO_EDGE,
                .wrapT = GL_CLAMP_TO_EDGE,
                .minFilter = GL_NEAREST,
                .magFilter = GL_NEAREST,
                .multiSampled = params.multiSampled,
                .numSamples = params.numSamples
            })
            , depthStencilBuffer({
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
                .numSamples = params.numSamples
            }) {
        framebuffer.bind();
        framebuffer.attachTexture(colorBuffer, GL_COLOR_ATTACHMENT0);
        framebuffer.attachTexture(normalsBuffer, GL_COLOR_ATTACHMENT1);
        framebuffer.attachTexture(idBuffer, GL_COLOR_ATTACHMENT2);
        framebuffer.attachTexture(depthStencilBuffer, GL_DEPTH_STENCIL_ATTACHMENT);

        uint attachments[3] = {
            GL_COLOR_ATTACHMENT0,
            GL_COLOR_ATTACHMENT1,
            GL_COLOR_ATTACHMENT2
        };
        glDrawBuffers(3, attachments);

        if (!framebuffer.checkStatus()) {
            throw std::runtime_error("FrameRenderTarget Framebuffer is not complete!");
        }

        framebuffer.unbind();
    }

    void blitToRenderTarget(RenderTarget& rt, GLenum filter = GL_NEAREST) {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer.ID);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, rt.getFramebufferID());

        // Color
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        GLenum drawBuffers[] = { GL_COLOR_ATTACHMENT0 };
        glDrawBuffers(1, drawBuffers);
        glBlitFramebuffer(0, 0, width, height,
                          0, 0, rt.width, rt.height,
                          GL_COLOR_BUFFER_BIT, filter);

        // Depth and stencil
        glBlitFramebuffer(0, 0, width, height,
                          0, 0, rt.width, rt.height,
                          GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT, GL_NEAREST);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void blitToFrameRT(FrameRenderTarget& frameRT, GLenum filter = GL_NEAREST) {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer.ID);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, frameRT.getFramebufferID());

        // Color
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        GLenum drawBuffers0[] = { GL_COLOR_ATTACHMENT0 };
        glDrawBuffers(1, drawBuffers0);
        glBlitFramebuffer(0, 0, width, height,
                          0, 0, frameRT.width, frameRT.height,
                          GL_COLOR_BUFFER_BIT, filter);

        // Normals
        glReadBuffer(GL_COLOR_ATTACHMENT1);
        GLenum drawBuffers1[] = { GL_COLOR_ATTACHMENT1 };
        glDrawBuffers(1, drawBuffers1);
        glBlitFramebuffer(0, 0, width, height,
                          0, 0, frameRT.width, frameRT.height,
                          GL_COLOR_BUFFER_BIT, filter);

        // Ids
        glReadBuffer(GL_COLOR_ATTACHMENT2);
        GLenum drawBuffers2[] = { GL_COLOR_ATTACHMENT2 };
        glDrawBuffers(1, drawBuffers2);
        glBlitFramebuffer(0, 0, width, height,
                          0, 0, frameRT.width, frameRT.height,
                          GL_COLOR_BUFFER_BIT, GL_NEAREST);

        // Depth and stencil
        glBlitFramebuffer(0, 0, width, height,
                          0, 0, frameRT.width, frameRT.height,
                          GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT, GL_NEAREST);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void blitToScreen(uint width, uint height) {
        framebuffer.blitToScreen(width, height);
    }

    void resize(uint width, uint height) override {
        RenderTargetBase::resize(width, height);

        colorBuffer.resize(width, height);
        normalsBuffer.resize(width, height);
        idBuffer.resize(width, height);
        depthStencilBuffer.resize(width, height);
    }

    void readPixels(unsigned char *data, bool readAsFloat = false) {
        bind();
        colorBuffer.readPixels(data, readAsFloat);
        unbind();
    }

    void saveColorAsPNG(const std::string& path) {
        bind();
        colorBuffer.saveAsPNG(path);
        unbind();
    }

    void saveColorAsJPG(const std::string& path, int quality = 95) {
        bind();
        colorBuffer.saveAsJPG(path, quality);
        unbind();
    }

    void saveColorAsHDR(const std::string& path) {
        bind();
        colorBuffer.saveAsHDR(path);
        unbind();
    }
};

} // namespace quasar

#endif // FRAME_RENDER_TARGET_H
