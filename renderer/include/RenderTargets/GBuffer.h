#ifndef GBUFFER_H
#define GBUFFER_H

#include <RenderTargets/RenderTargetBase.h>

class GeometryBuffer : public RenderTargetBase {
public:
    Texture positionBuffer;
    Texture normalsBuffer;
    Texture colorBuffer;
    Texture depthBuffer;

    explicit GeometryBuffer(const RenderTargetCreateParams &params)
            : RenderTargetBase(params),
            positionBuffer({ .width = width,
                             .height = height,
                             .internalFormat = GL_RGBA16F,
                             .format = GL_RGBA,
                             .type = GL_FLOAT,
                             .wrapS = GL_CLAMP_TO_EDGE,
                             .wrapT = GL_CLAMP_TO_EDGE,
                             .minFilter = GL_NEAREST,
                             .magFilter = GL_NEAREST }),
            normalsBuffer({  .width = width,
                            .height = height,
                            .internalFormat = GL_RGBA16F,
                            .format = GL_RGBA,
                            .type = GL_FLOAT,
                            .wrapS = GL_CLAMP_TO_EDGE,
                            .wrapT = GL_CLAMP_TO_EDGE,
                            .minFilter = GL_NEAREST,
                            .magFilter = GL_NEAREST }),
            colorBuffer({ .width = width,
                          .height = height,
                          .internalFormat = GL_RGBA16F,
                          .format = GL_RGBA,
                          .type = GL_FLOAT,
                          .wrapS = GL_CLAMP_TO_EDGE,
                          .wrapT = GL_CLAMP_TO_EDGE,
                          .minFilter = GL_LINEAR,
                          .magFilter = GL_LINEAR }),
            depthBuffer(  { .width = width,
                          .height = height,
                          .internalFormat = GL_DEPTH_COMPONENT32F,
                          .format = GL_DEPTH_COMPONENT,
                          .type = GL_FLOAT,
                          .minFilter = GL_NEAREST,
                          .magFilter = GL_NEAREST }) {

        framebuffer.bind();
        framebuffer.attachTexture(positionBuffer, GL_COLOR_ATTACHMENT0);
        framebuffer.attachTexture(normalsBuffer, GL_COLOR_ATTACHMENT1);
        framebuffer.attachTexture(colorBuffer, GL_COLOR_ATTACHMENT2);
        framebuffer.attachTexture(depthBuffer, GL_DEPTH_ATTACHMENT);

        unsigned int attachments[3] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2};
        glDrawBuffers(3, attachments);

        if (!framebuffer.checkStatus()) {
            throw std::runtime_error("GBuffer Framebuffer is not complete!");
        }

        framebuffer.unbind();
    }

    void resize(unsigned int width, unsigned int height) override {
        this->width = width;
        this->height = height;

        positionBuffer.resize(width, height);
        normalsBuffer.resize(width, height);
        colorBuffer.resize(width, height);
        depthBuffer.resize(width, height);
    }
};

#endif // GBUFFER_H
