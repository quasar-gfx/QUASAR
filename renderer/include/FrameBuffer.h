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

        colorBuffer = Texture(width, height, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE);
        depthBuffer = Texture(width, height, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, GL_FLOAT);

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

struct DirShadowBuffer : public FrameBuffer {
public:
    void createColorAndDepthBuffers(unsigned int width, unsigned int height) {
        this->width = width;
        this->height = height;

        glGenFramebuffers(1, &ID);
        glBindFramebuffer(GL_FRAMEBUFFER, ID);

        depthBuffer = Texture(width, height, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, GL_FLOAT,
                                GL_CLAMP_TO_BORDER, GL_CLAMP_TO_BORDER,
                                GL_NEAREST, GL_NEAREST,
                                nullptr, true);

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthBuffer.ID, 0);

        glDrawBuffer(GL_NONE);
        glReadBuffer(GL_NONE);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            throw std::runtime_error("DirShadowBuffer Framebuffer is not complete!");
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
};

struct PointShadowBuffer : public FrameBuffer {
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
