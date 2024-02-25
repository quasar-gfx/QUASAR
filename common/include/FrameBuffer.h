#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H

#include "glad/glad.h"

#include "OpenGLObject.h"
#include "Texture.h"
#include "RenderBuffer.h"

class FrameBuffer : public OpenGLObject {
public:
    unsigned int width, height;

    Texture* colorAttachment;
    RenderBuffer* depthAttachment;

    FrameBuffer(unsigned int width, unsigned int height)
            : width(width), height(height) {
        colorAttachment = Texture::create(width, height, TEXTURE_DIFFUSE, GL_RGB, GL_CLAMP_TO_EDGE, GL_LINEAR);
        depthAttachment = RenderBuffer::create(width, height, GL_DEPTH24_STENCIL8);

        glGenFramebuffers(1, &ID);
        glBindFramebuffer(GL_FRAMEBUFFER, ID);

        // create a color attachment texture
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorAttachment->ID, 0);

        // create a renderbuffer object for depth and stencil attachment (we won't be sampling these)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, depthAttachment->ID);

        // now that we actually created the framebuffer and added all attachments we want to check if it is actually complete now
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

    void bindColorAttachment(unsigned int slot) {
        colorAttachment->bind(slot);
    }

    void unbindColorAttachment() {
        colorAttachment->unbind();
    }

    void bindDepthAttachment() {
        depthAttachment->bind();
    }

    void unbindDepthAttachment() {
        depthAttachment->unbind();
    }

    void unbind() {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void cleanup() {
        glDeleteFramebuffers(1, &ID);
        colorAttachment->cleanup();
        depthAttachment->cleanup();
    }
};

#endif // FRAMEBUFFER_H
