#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H

#include <glad/glad.h>

#include <OpenGLObject.h>
#include <Texture.h>
#include <CubeMap.h>

class Framebuffer : public OpenGLObject {
public:
    unsigned int width, height;

    Texture colorBuffer;
    Texture depthBuffer;

    explicit Framebuffer() = default;

    void createColorAndDepthBuffers(unsigned int width, unsigned int height) {
        this->width = width;
        this->height = height;

        glGenFramebuffers(1, &ID);
        glBindFramebuffer(GL_FRAMEBUFFER, ID);

        colorBuffer = Texture({
            .width = width,
            .height = height,
            .internalFormat = GL_RGBA16,
            .format = GL_RGBA,
            .type = GL_FLOAT,
            .wrapS = GL_CLAMP_TO_EDGE,
            .wrapT = GL_CLAMP_TO_EDGE,
            .minFilter = GL_LINEAR,
            .magFilter = GL_LINEAR
        });
        depthBuffer = Texture({
            .width = width,
            .height = height,
            .internalFormat = GL_DEPTH_COMPONENT24,
            .format = GL_DEPTH_COMPONENT,
            .type = GL_FLOAT,
            .minFilter = GL_NEAREST,
            .magFilter = GL_NEAREST
        });

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorBuffer.ID, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthBuffer.ID, 0);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            throw std::runtime_error("Framebuffer is not complete!");
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    ~Framebuffer() {
        cleanup();
    }

    void resize(unsigned int width, unsigned int height) {
        this->width = width;
        this->height = height;

        colorBuffer.resize(width, height);
        depthBuffer.resize(width, height);
    }

    void bind() {
        glBindFramebuffer(GL_FRAMEBUFFER, ID);
        glViewport(0, 0, width, height);
    }

    void unbind() {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void cleanup() {
        glDeleteFramebuffers(1, &ID);
    }
};

class GeometryBuffer : public Framebuffer {
public:
    Texture positionBuffer;
    Texture normalsBuffer;

    void createBuffers(unsigned int width, unsigned int height) {
        this->width = width;
        this->height = height;

        glGenFramebuffers(1, &ID);
        glBindFramebuffer(GL_FRAMEBUFFER, ID);

        positionBuffer = Texture({
            .width = width,
            .height = height,
            .internalFormat = GL_RGBA16F,
            .format = GL_RGBA,
            .type = GL_FLOAT,
            .wrapS = GL_CLAMP_TO_EDGE,
            .wrapT = GL_CLAMP_TO_EDGE,
            .minFilter = GL_NEAREST,
            .magFilter = GL_NEAREST
        });
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, positionBuffer.ID, 0);

        normalsBuffer = Texture({
            .width = width,
            .height = height,
            .internalFormat = GL_RGBA16F,
            .format = GL_RGBA,
            .type = GL_FLOAT,
            .wrapS = GL_CLAMP_TO_EDGE,
            .wrapT = GL_CLAMP_TO_EDGE,
            .minFilter = GL_NEAREST,
            .magFilter = GL_NEAREST
        });
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, normalsBuffer.ID, 0);

        colorBuffer = Texture({
            .width = width,
            .height = height,
            .internalFormat = GL_RGBA16,
            .format = GL_RGBA,
            .type = GL_FLOAT,
            .wrapS = GL_CLAMP_TO_EDGE,
            .wrapT = GL_CLAMP_TO_EDGE,
            .minFilter = GL_LINEAR,
            .magFilter = GL_LINEAR
        });
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, colorBuffer.ID, 0);

        depthBuffer = Texture({
            .width = width,
            .height = height,
            .internalFormat = GL_DEPTH_COMPONENT24,
            .format = GL_DEPTH_COMPONENT,
            .type = GL_FLOAT,
            .minFilter = GL_NEAREST,
            .magFilter = GL_NEAREST
        });
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthBuffer.ID, 0);

        unsigned int attachments[3] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2};
        glDrawBuffers(3, attachments);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            throw std::runtime_error("GBuffer Framebuffer is not complete!");
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void resize(unsigned int width, unsigned int height) {
        this->width = width;
        this->height = height;

        positionBuffer.resize(width, height);
        normalsBuffer.resize(width, height);
        colorBuffer.resize(width, height);
        depthBuffer.resize(width, height);
    }
};

class DirLightShadowBuffer : public Framebuffer {
public:
    void createColorAndDepthBuffers(unsigned int width, unsigned int height) {
        this->width = width;
        this->height = height;

        glGenFramebuffers(1, &ID);
        glBindFramebuffer(GL_FRAMEBUFFER, ID);

        depthBuffer = Texture({
            .width = width,
            .height = height,
            .internalFormat = GL_DEPTH_COMPONENT,
            .format = GL_DEPTH_COMPONENT,
            .type = GL_FLOAT,
            .wrapS = GL_CLAMP_TO_BORDER,
            .wrapT = GL_CLAMP_TO_BORDER,
            .minFilter = GL_NEAREST,
            .magFilter = GL_NEAREST,
            .hasBorder = true
        });
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthBuffer.ID, 0);

        glDrawBuffer(GL_NONE);
        glReadBuffer(GL_NONE);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            throw std::runtime_error("DirLightShadowBuffer Framebuffer is not complete!");
        }

        glClear(GL_DEPTH_BUFFER_BIT);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
};

class PointLightShadowBuffer : public Framebuffer {
public:
    CubeMap depthCubeMap;

    void createColorAndDepthBuffers(unsigned int width, unsigned int height) {
        this->width = width;
        this->height = height;

        glGenFramebuffers(1, &ID);
        glBindFramebuffer(GL_FRAMEBUFFER, ID);

        depthCubeMap.init(width, height, CubeMapType::SHADOW);
        glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depthCubeMap.ID, 0);

        glDrawBuffer(GL_NONE);
        glReadBuffer(GL_NONE);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            throw std::runtime_error("PointLightShadowBuffer Framebuffer is not complete!");
        }

        glClear(GL_DEPTH_BUFFER_BIT);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
};

#endif // FRAMEBUFFER_H
