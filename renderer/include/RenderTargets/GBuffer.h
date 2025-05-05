#ifndef GBUFFER_H
#define GBUFFER_H

#include <RenderTargets/RenderTargetBase.h>
#include <RenderTargets/RenderTarget.h>
#include <RenderTargets/FrameRenderTarget.h>

namespace quasar {

class GBuffer : public RenderTargetBase {
public:
    Texture albedoBuffer;
    Texture pbrBuffer;
    Texture alphaBuffer;
    Texture positionBuffer;
    Texture normalsBuffer;
    Texture lightPositionBuffer;
    Texture idBuffer;
    Texture depthStencilBuffer;

    GBuffer(const RenderTargetCreateParams &params)
            : RenderTargetBase(params)
            , albedoBuffer({
                .width = width,
                .height = height,
                .internalFormat = GL_RGBA8,
                .format = GL_RGBA,
                .type = GL_UNSIGNED_BYTE,
                .wrapS = GL_CLAMP_TO_EDGE,
                .wrapT = GL_CLAMP_TO_EDGE,
                .minFilter = params.minFilter,
                .magFilter = params.magFilter,
                .multiSampled = params.multiSampled,
                .numSamples = params.numSamples
            })
            , pbrBuffer({
                .width = width,
                .height = height,
                .internalFormat = GL_RGBA8,
                .format = GL_RGBA,
                .type = GL_UNSIGNED_BYTE,
                .wrapS = GL_CLAMP_TO_EDGE,
                .wrapT = GL_CLAMP_TO_EDGE,
                .minFilter = params.minFilter,
                .magFilter = params.magFilter,
                .multiSampled = params.multiSampled,
                .numSamples = params.numSamples
            })
            , alphaBuffer({
                .width = width,
                .height = height,
                .internalFormat = GL_RG8,
                .format = GL_RG,
                .type = GL_UNSIGNED_BYTE,
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
                .minFilter = params.minFilter,
                .magFilter = params.magFilter,
                .multiSampled = params.multiSampled,
                .numSamples = params.numSamples
            })
            , positionBuffer({
                .width = width,
                .height = height,
                .internalFormat = GL_RGBA16F,
                .format = GL_RGBA,
                .type = GL_HALF_FLOAT,
                .wrapS = GL_CLAMP_TO_EDGE,
                .wrapT = GL_CLAMP_TO_EDGE,
                .minFilter = GL_NEAREST,
                .magFilter = GL_NEAREST,
                .multiSampled = params.multiSampled,
                .numSamples = params.numSamples
            })
            , lightPositionBuffer({
                .width = width,
                .height = height,
                .internalFormat = GL_RGBA16F,
                .format = GL_RGBA,
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
        framebuffer.attachTexture(albedoBuffer, GL_COLOR_ATTACHMENT0);
        framebuffer.attachTexture(pbrBuffer, GL_COLOR_ATTACHMENT1);
        framebuffer.attachTexture(alphaBuffer, GL_COLOR_ATTACHMENT2);
        framebuffer.attachTexture(normalsBuffer, GL_COLOR_ATTACHMENT3);
        framebuffer.attachTexture(positionBuffer, GL_COLOR_ATTACHMENT4);
        framebuffer.attachTexture(lightPositionBuffer, GL_COLOR_ATTACHMENT5);
        framebuffer.attachTexture(idBuffer, GL_COLOR_ATTACHMENT6);
        framebuffer.attachTexture(depthStencilBuffer, GL_DEPTH_STENCIL_ATTACHMENT);

        uint attachments[7] = {
            GL_COLOR_ATTACHMENT0,
            GL_COLOR_ATTACHMENT1,
            GL_COLOR_ATTACHMENT2,
            GL_COLOR_ATTACHMENT3,
            GL_COLOR_ATTACHMENT4,
            GL_COLOR_ATTACHMENT5,
            GL_COLOR_ATTACHMENT6
        };
        glDrawBuffers(7, attachments);

        if (!framebuffer.checkStatus()) {
            throw std::runtime_error("FrameRenderTarget Framebuffer is not complete!");
        }

        framebuffer.unbind();
    }

    void blitToFrameRT(FrameRenderTarget & frameRT, GLenum filter = GL_NEAREST) {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer.ID);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, frameRT.getFramebufferID());

        // normals
        glReadBuffer(GL_COLOR_ATTACHMENT3);
        GLenum drawBuffers1[] = { GL_COLOR_ATTACHMENT1 };
        glDrawBuffers(1, drawBuffers1);
        glBlitFramebuffer(0, 0, width, height,
                          0, 0, frameRT.width, frameRT.height,
                          GL_COLOR_BUFFER_BIT, filter);

        // id buffer
        glReadBuffer(GL_COLOR_ATTACHMENT6);
        GLenum drawBuffers2[] = { GL_COLOR_ATTACHMENT2 };
        glDrawBuffers(1, drawBuffers2);
        glBlitFramebuffer(0, 0, width, height,
                          0, 0, frameRT.width, frameRT.height,
                          GL_COLOR_BUFFER_BIT, GL_NEAREST);

        // depth and stencil
        glBlitFramebuffer(0, 0, width, height,
                          0, 0, frameRT.width, frameRT.height,
                          GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT, GL_NEAREST);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void blitToFrameRT(GBuffer& gBuffer, GLenum filter = GL_NEAREST) {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer.ID);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, gBuffer.getFramebufferID());

        // colors
        GLenum drawBuffers[] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2,
                                 GL_COLOR_ATTACHMENT3, GL_COLOR_ATTACHMENT4, GL_COLOR_ATTACHMENT5,
                                 GL_COLOR_ATTACHMENT6 };
        glDrawBuffers(7, drawBuffers);

        for (int i = 0; i < 7; ++i) {
            glReadBuffer(GL_COLOR_ATTACHMENT0 + i);
            glBlitFramebuffer(0, 0, width, height,
                              0, 0, gBuffer.width, gBuffer.height,
                              GL_COLOR_BUFFER_BIT, filter);
        }

        // depth and stencil
        glBlitFramebuffer(0, 0, width, height,
                          0, 0, gBuffer.width, gBuffer.height,
                          GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT, GL_NEAREST);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void blitDepthToRenderTarget(RenderTarget &renderTarget) {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer.ID);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, renderTarget.getFramebufferID());

        glBlitFramebuffer(0, 0, width, height,
                          0, 0, renderTarget.width, renderTarget.height,
                          GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT, GL_NEAREST);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void blitToScreen(uint width, uint height) {
        framebuffer.blitToScreen(width, height);
    }

    void resize(uint width, uint height) override {
        RenderTargetBase::resize(width, height);

        albedoBuffer.resize(width, height);
        pbrBuffer.resize(width, height);
        alphaBuffer.resize(width, height);
        positionBuffer.resize(width, height);
        normalsBuffer.resize(width, height);
        lightPositionBuffer.resize(width, height);
        idBuffer.resize(width, height);
        depthStencilBuffer.resize(width, height);
    }

    void readPixels(unsigned char *data, bool readAsFloat = false) {
        bind();
        albedoBuffer.readPixels(data, readAsFloat);
        unbind();
    }

    void saveColorAsPNG(const std::string &path) {
        bind();
        albedoBuffer.saveAsPNG(path);
        unbind();
    }

    void saveColorAsHDR(const std::string &path) {
        bind();
        albedoBuffer.saveAsHDR(path);
        unbind();
    }
};

} // namespace quasar

#endif // GBUFFER_H
