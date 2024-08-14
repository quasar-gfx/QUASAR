#ifndef GBUFFER_H
#define GBUFFER_H

#include <RenderTargets/RenderTargetBase.h>

class GeometryBuffer : public RenderTargetBase {
public:
    Texture positionBuffer;
    Texture normalsBuffer;
    Texture idBuffer;
    Texture colorBuffer;
    Texture depthBuffer;

    explicit GeometryBuffer(const RenderTargetCreateParams &params)
            : RenderTargetBase(params)
            , colorBuffer({
                .width = width,
                .height = height,
                .internalFormat = GL_RGBA16F,
                .format = GL_RGBA,
                .type = GL_FLOAT,
                .wrapS = GL_CLAMP_TO_EDGE,
                .wrapT = GL_CLAMP_TO_EDGE,
                .minFilter = GL_LINEAR,
                .magFilter = GL_LINEAR,
                .multiSampled = params.multiSampled
            })
            , depthBuffer({
                .width = width,
                .height = height,
                .internalFormat = GL_DEPTH24_STENCIL8,
                .format = GL_DEPTH_STENCIL,
                .type = GL_UNSIGNED_INT_24_8,
                .minFilter = GL_NEAREST,
                .magFilter = GL_NEAREST,
                .multiSampled = params.multiSampled
            })
            , positionBuffer({
                .width = width,
                .height = height,
                .internalFormat = GL_RGB16F,
                .format = GL_RGB,
                .type = GL_FLOAT,
                .wrapS = GL_CLAMP_TO_EDGE,
                .wrapT = GL_CLAMP_TO_EDGE,
                .minFilter = GL_NEAREST,
                .magFilter = GL_NEAREST ,
                .multiSampled = params.multiSampled
            })
            , normalsBuffer({
                .width = width,
                .height = height,
                .internalFormat = GL_RGB16F,
                .format = GL_RGB,
                .type = GL_FLOAT,
                .wrapS = GL_CLAMP_TO_EDGE,
                .wrapT = GL_CLAMP_TO_EDGE,
                .minFilter = GL_NEAREST,
                .magFilter = GL_NEAREST ,
                .multiSampled = params.multiSampled
            })
            , idBuffer({
                .width = width,
                .height = height,
                .internalFormat = GL_R32UI,
                .format = GL_RED_INTEGER,
                .type = GL_UNSIGNED_INT,
                .wrapS = GL_CLAMP_TO_EDGE,
                .wrapT = GL_CLAMP_TO_EDGE,
                .minFilter = GL_NEAREST,
                .magFilter = GL_NEAREST,
                .multiSampled = params.multiSampled
            }) {

        framebuffer.bind();
        framebuffer.attachTexture(colorBuffer, GL_COLOR_ATTACHMENT0);
        framebuffer.attachTexture(positionBuffer, GL_COLOR_ATTACHMENT1);
        framebuffer.attachTexture(normalsBuffer, GL_COLOR_ATTACHMENT2);
        framebuffer.attachTexture(idBuffer, GL_COLOR_ATTACHMENT3);
        framebuffer.attachTexture(depthBuffer, GL_DEPTH_STENCIL_ATTACHMENT);

        unsigned int attachments[4] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3 };
        glDrawBuffers(4, attachments);

        if (!framebuffer.checkStatus()) {
            throw std::runtime_error("GBuffer Framebuffer is not complete!");
        }

        framebuffer.unbind();
    }

    void blitToGBuffer(GeometryBuffer &gBuffer) {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer.ID);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, gBuffer.framebuffer.ID);

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

        glBlitFramebuffer(0, 0, width, height, 0, 0, gBuffer.width, gBuffer.height, GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT, GL_NEAREST);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void resize(unsigned int width, unsigned int height) override {
        this->width = width;
        this->height = height;

        positionBuffer.resize(width, height);
        normalsBuffer.resize(width, height);
        idBuffer.resize(width, height);
        colorBuffer.resize(width, height);
        depthBuffer.resize(width, height);
    }

    void saveColorAsPNG(const std::string &path) {
        colorBuffer.saveAsPNG(path);
    }

    void saveColorAsHDR(const std::string &path) {
        colorBuffer.saveAsHDR(path);
    }
};

#endif // GBUFFER_H
