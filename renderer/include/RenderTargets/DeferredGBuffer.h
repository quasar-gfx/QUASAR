#ifndef DEFERRED_GBUFFER_H
#define DEFERRED_GBUFFER_H

#include <RenderTargets/RenderTargetBase.h>
#include <RenderTargets/RenderTarget.h>

class DeferredGBuffer : public RenderTargetBase {
public:
    Texture albedoBuffer;
    Texture pbrBuffer;
    Texture emissiveBuffer;
    Texture lightPositionXYZBuffer;
    Texture lightPositionWIBLAlphaBuffer;
    Texture positionBuffer;
    Texture normalsBuffer;
    Texture idBuffer;
    Texture depthStencilBuffer;

    DeferredGBuffer(const RenderTargetCreateParams &params)
            : RenderTargetBase(params)
            , albedoBuffer({
                .width = width,
                .height = height,
                .internalFormat = GL_RGBA16F,
                .format = GL_RGB,
                .type = GL_HALF_FLOAT,
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
                .internalFormat = GL_RGBA16F,
                .format = GL_RGB,
                .type = GL_HALF_FLOAT,
                .wrapS = GL_CLAMP_TO_EDGE,
                .wrapT = GL_CLAMP_TO_EDGE,
                .minFilter = params.minFilter,
                .magFilter = params.magFilter,
                .multiSampled = params.multiSampled,
                .numSamples = params.numSamples
            })
            , emissiveBuffer({
                .width = width,
                .height = height,
                .internalFormat = GL_RGBA16F,
                .format = GL_RGB,
                .type = GL_HALF_FLOAT,
                .wrapS = GL_CLAMP_TO_EDGE,
                .wrapT = GL_CLAMP_TO_EDGE,
                .minFilter = params.minFilter,
                .magFilter = params.magFilter,
                .multiSampled = params.multiSampled,
                .numSamples = params.numSamples
            })
            , lightPositionXYZBuffer({
                .width = width,
                .height = height,
                .internalFormat = GL_RGB32F,
                .format = GL_RGB,
                .type = GL_FLOAT,
                .wrapS = GL_CLAMP_TO_EDGE,
                .wrapT = GL_CLAMP_TO_EDGE,
                .minFilter = params.minFilter,
                .magFilter = params.magFilter,
                .multiSampled = params.multiSampled,
                .numSamples = params.numSamples
            })
            , lightPositionWIBLAlphaBuffer({
                .width = width,
                .height = height,
                .internalFormat = GL_RGB32F,
                .format = GL_RGB,
                .type = GL_FLOAT,
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
                .internalFormat = GL_RGB32F,
                .format = GL_RGB,
                .type = GL_FLOAT,
                .wrapS = GL_CLAMP_TO_EDGE,
                .wrapT = GL_CLAMP_TO_EDGE,
                .minFilter = GL_NEAREST,
                .magFilter = GL_NEAREST ,
                .multiSampled = params.multiSampled,
                .numSamples = params.numSamples
            })
            , normalsBuffer({
                .width = width,
                .height = height,
                .internalFormat = GL_RGB32F,
                .format = GL_RGB,
                .type = GL_FLOAT,
                .wrapS = GL_CLAMP_TO_EDGE,
                .wrapT = GL_CLAMP_TO_EDGE,
                .minFilter = GL_NEAREST,
                .magFilter = GL_NEAREST ,
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
                .internalFormat = GL_DEPTH24_STENCIL8,
                .format = GL_DEPTH_STENCIL,
                .type = GL_UNSIGNED_INT_24_8,
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
        framebuffer.attachTexture(emissiveBuffer, GL_COLOR_ATTACHMENT2);
        framebuffer.attachTexture(lightPositionXYZBuffer, GL_COLOR_ATTACHMENT3);
        framebuffer.attachTexture(lightPositionWIBLAlphaBuffer, GL_COLOR_ATTACHMENT4);
        framebuffer.attachTexture(positionBuffer, GL_COLOR_ATTACHMENT5);
        framebuffer.attachTexture(normalsBuffer, GL_COLOR_ATTACHMENT6);
        framebuffer.attachTexture(idBuffer, GL_COLOR_ATTACHMENT7);
        framebuffer.attachTexture(depthStencilBuffer, GL_DEPTH_STENCIL_ATTACHMENT);

        unsigned int attachments[8] = {
            GL_COLOR_ATTACHMENT0,
            GL_COLOR_ATTACHMENT1,
            GL_COLOR_ATTACHMENT2,
            GL_COLOR_ATTACHMENT3,
            GL_COLOR_ATTACHMENT4,
            GL_COLOR_ATTACHMENT5,
            GL_COLOR_ATTACHMENT6,
            GL_COLOR_ATTACHMENT7
        };
        glDrawBuffers(8, attachments);

        if (!framebuffer.checkStatus()) {
            throw std::runtime_error("GBuffer Framebuffer is not complete!");
        }

        framebuffer.unbind();
    }

#ifdef GL_CORE
    void blitToGBuffer(DeferredGBuffer &gBuffer) {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer.ID);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, gBuffer.getFramebufferID());

        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glDrawBuffer(GL_COLOR_ATTACHMENT0);
        glBlitFramebuffer(0, 0, width, height, 0, 0, gBuffer.width, gBuffer.height, GL_COLOR_BUFFER_BIT, GL_NEAREST);

        glReadBuffer(GL_COLOR_ATTACHMENT1);
        glDrawBuffer(GL_COLOR_ATTACHMENT1);
        glBlitFramebuffer(0, 0, width, height, 0, 0, gBuffer.width, gBuffer.height, GL_COLOR_BUFFER_BIT, GL_NEAREST);

        glReadBuffer(GL_COLOR_ATTACHMENT2);
        glDrawBuffer(GL_COLOR_ATTACHMENT2);
        glBlitFramebuffer(0, 0, width, height, 0, 0, gBuffer.width, gBuffer.height, GL_COLOR_BUFFER_BIT, GL_NEAREST);

        glReadBuffer(GL_COLOR_ATTACHMENT3);
        glDrawBuffer(GL_COLOR_ATTACHMENT3);
        glBlitFramebuffer(0, 0, width, height, 0, 0, gBuffer.width, gBuffer.height, GL_COLOR_BUFFER_BIT, GL_NEAREST);

        glReadBuffer(GL_COLOR_ATTACHMENT4);
        glDrawBuffer(GL_COLOR_ATTACHMENT4);
        glBlitFramebuffer(0, 0, width, height, 0, 0, gBuffer.width, gBuffer.height, GL_COLOR_BUFFER_BIT, GL_NEAREST);

        glReadBuffer(GL_COLOR_ATTACHMENT5);
        glDrawBuffer(GL_COLOR_ATTACHMENT5);
        glBlitFramebuffer(0, 0, width, height, 0, 0, gBuffer.width, gBuffer.height, GL_COLOR_BUFFER_BIT, GL_NEAREST);

        glReadBuffer(GL_COLOR_ATTACHMENT6);
        glDrawBuffer(GL_COLOR_ATTACHMENT6);
        glBlitFramebuffer(0, 0, width, height, 0, 0, gBuffer.width, gBuffer.height, GL_COLOR_BUFFER_BIT, GL_NEAREST);

        glReadBuffer(GL_COLOR_ATTACHMENT7);
        glDrawBuffer(GL_COLOR_ATTACHMENT7);
        glBlitFramebuffer(0, 0, width, height, 0, 0, gBuffer.width, gBuffer.height, GL_COLOR_BUFFER_BIT, GL_NEAREST);

        glBlitFramebuffer(0, 0, width, height, 0, 0, gBuffer.width, gBuffer.height, GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT, GL_NEAREST);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void blitToRenderTarget(RenderTarget &renderTarget) {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer.ID);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, renderTarget.getFramebufferID());

        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glDrawBuffer(GL_COLOR_ATTACHMENT0);
        glBlitFramebuffer(0, 0, width, height, 0, 0, renderTarget.width, renderTarget.height, GL_COLOR_BUFFER_BIT, GL_NEAREST);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void blitDepthToRenderTarget(RenderTarget &renderTarget) {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer.ID);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, renderTarget.getFramebufferID());

        glBlitFramebuffer(0, 0, width, height, 0, 0, renderTarget.width, renderTarget.height, GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT, GL_NEAREST);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void blitToScreen(unsigned int width, unsigned int height) {
        framebuffer.blitToScreen(width, height);
    }
#endif

    void resize(unsigned int width, unsigned int height) override {
        RenderTargetBase::resize(width, height);

        albedoBuffer.resize(width, height);
        pbrBuffer.resize(width, height);
        emissiveBuffer.resize(width, height);
        lightPositionXYZBuffer.resize(width, height);
        lightPositionWIBLAlphaBuffer.resize(width, height);
        positionBuffer.resize(width, height);
        normalsBuffer.resize(width, height);
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

#endif // DEFERRED_GBUFFER_H
