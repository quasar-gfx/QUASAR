#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H

#include <glad/glad.h>

#include <OpenGLObject.h>
#include <Texture.h>
#include <RenderBuffer.h>

class FrameBuffer : public OpenGLObject {
public:
    unsigned int width, height;

    Texture colorBuffer;
    Texture depthBuffer;

    FrameBuffer(unsigned int width, unsigned int height)
            : width(width), height(height),
              colorBuffer(width, height, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE),
              depthBuffer(width, height, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, GL_FLOAT){
        glGenFramebuffers(1, &ID);
        glBindFramebuffer(GL_FRAMEBUFFER, ID);

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
    }

    void unbind() {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void attachStencilBuffer(RenderBuffer &stencilBuffer) {
        bind();
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_TEXTURE_2D, stencilBuffer.ID);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            throw std::runtime_error("Framebuffer is not complete!");
        }

        unbind();
    }

    void unattachStencilBuffer() {
        bind();
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_TEXTURE_2D, 0);
        unbind();
    }

    void cleanup() {
        glDeleteFramebuffers(1, &ID);
    }
};

#endif // FRAMEBUFFER_H
