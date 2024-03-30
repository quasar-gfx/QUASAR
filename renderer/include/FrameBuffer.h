#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H

#include <glad/glad.h>

#include <OpenGLObject.h>
#include <Texture.h>
#include <CubeMap.h>

class FrameBuffer : public OpenGLObject {
public:
    unsigned int width, height;

    Texture colorBuffer;
    Texture depthBuffer;

    FrameBuffer() = default;

    void createColorAndDepthBuffers(unsigned int width, unsigned int height) {
        this->width = width;
        this->height = height;

        glGenFramebuffers(1, &ID);
        glBindFramebuffer(GL_FRAMEBUFFER, ID);

        TextureCreateParams colorParams{
            .width = width,
            .height = height,
            .internalFormat = GL_RGBA16,
            .format = GL_RGBA,
            .type = GL_FLOAT,
            .wrapS = GL_CLAMP_TO_EDGE,
            .wrapT = GL_CLAMP_TO_EDGE,
            .minFilter = GL_LINEAR,
            .magFilter = GL_LINEAR
        };
        colorBuffer = Texture(colorParams);
        TextureCreateParams depthParams{
            .width = width,
            .height = height,
            .internalFormat = GL_DEPTH_COMPONENT24,
            .format = GL_DEPTH_COMPONENT,
            .type = GL_FLOAT
        };
        depthBuffer = Texture(depthParams);

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorBuffer.ID, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthBuffer.ID, 0);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            throw std::runtime_error("Framebuffer is not complete!");
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    ~FrameBuffer() {
        cleanup();
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

class GeometryBuffer : public FrameBuffer {
public:
    Texture positionBuffer;
    Texture normalsBuffer;

    void createBuffers(unsigned int width, unsigned int height) {
        this->width = width;
        this->height = height;

        glGenFramebuffers(1, &ID);
        glBindFramebuffer(GL_FRAMEBUFFER, ID);

        TextureCreateParams positionParams{
            .width = width,
            .height = height,
            .internalFormat = GL_RGBA16F,
            .format = GL_RGBA,
            .type = GL_FLOAT,
            .wrapS = GL_CLAMP_TO_EDGE,
            .wrapT = GL_CLAMP_TO_EDGE,
            .minFilter = GL_NEAREST,
            .magFilter = GL_NEAREST
        };
        positionBuffer = Texture(positionParams);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, positionBuffer.ID, 0);

        TextureCreateParams normalsParams{
            .width = width,
            .height = height,
            .internalFormat = GL_RGBA16F,
            .format = GL_RGBA,
            .type = GL_FLOAT,
            .wrapS = GL_CLAMP_TO_EDGE,
            .wrapT = GL_CLAMP_TO_EDGE,
            .minFilter = GL_NEAREST,
            .magFilter = GL_NEAREST
        };
        normalsBuffer = Texture(normalsParams);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, normalsBuffer.ID, 0);

        TextureCreateParams colorParams{
            .width = width,
            .height = height,
            .internalFormat = GL_RGBA16,
            .format = GL_RGBA,
            .type = GL_FLOAT,
            .wrapS = GL_CLAMP_TO_EDGE,
            .wrapT = GL_CLAMP_TO_EDGE,
            .minFilter = GL_NEAREST,
            .magFilter = GL_NEAREST
        };
        colorBuffer = Texture(colorParams);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, colorBuffer.ID, 0);

        TextureCreateParams depthParams{
            .width = width,
            .height = height,
            .internalFormat = GL_DEPTH_COMPONENT24,
            .format = GL_DEPTH_COMPONENT,
            .type = GL_FLOAT
        };
        depthBuffer = Texture(depthParams);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthBuffer.ID, 0);

        unsigned int attachments[3] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2};
        glDrawBuffers(3, attachments);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            throw std::runtime_error("Framebuffer is not complete!");
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
};

class DirShadowBuffer : public FrameBuffer {
public:
    void createColorAndDepthBuffers(unsigned int width, unsigned int height) {
        this->width = width;
        this->height = height;

        glGenFramebuffers(1, &ID);
        glBindFramebuffer(GL_FRAMEBUFFER, ID);

        TextureCreateParams params{
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
        };
        depthBuffer = Texture(params);

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthBuffer.ID, 0);

        glDrawBuffer(GL_NONE);
        glReadBuffer(GL_NONE);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            throw std::runtime_error("DirShadowBuffer Framebuffer is not complete!");
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
};

class PointShadowBuffer : public FrameBuffer {
public:
    CubeMap depthCubeMap;

    void createColorAndDepthBuffers(unsigned int width, unsigned int height) {
        this->width = width;
        this->height = height;

        glGenFramebuffers(1, &ID);
        glBindFramebuffer(GL_FRAMEBUFFER, ID);

        depthCubeMap.init(width, height, CUBE_MAP_SHADOW);

        glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depthCubeMap.ID, 0);

        glDrawBuffer(GL_NONE);
        glReadBuffer(GL_NONE);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            throw std::runtime_error("PointShadowBuffer Framebuffer is not complete!");
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
};

#endif // FRAMEBUFFER_H
