#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H

#include <glad/glad.h>

#include <OpenGLObject.h>
#include <Texture.h>
#include <RenderBuffer.h>

class FrameBuffer : public OpenGLObject {
public:
    unsigned int width, height;

    Texture* colorAttachment;
    Texture* depthAttachment;

    ~FrameBuffer() {
        cleanup();
    }

    void bind() {
        glBindFramebuffer(GL_FRAMEBUFFER, ID);
    }

    void bindColorAttachment(unsigned int slot = 0) {
        colorAttachment->bind(slot);
    }

    void unbindColorAttachment() {
        colorAttachment->unbind();
    }

    void bindDepthAttachment(unsigned int slot = 0) {
        depthAttachment->bind(slot);
    }

    void unbindDepthAttachment() {
        depthAttachment->unbind();
    }

    void unbind() {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void cleanup() {
        glDeleteFramebuffers(1, &ID);
    }

    static FrameBuffer* create(unsigned int width, unsigned int height) {
        return new FrameBuffer(width, height);
    }

private:
    FrameBuffer(unsigned int width, unsigned int height)
            : width(width), height(height) {

        glGenFramebuffers(1, &ID);
        glBindFramebuffer(GL_FRAMEBUFFER, ID);

        // create a color attachment texture
        colorAttachment = Texture::create(width, height, TEXTURE_DIFFUSE, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE);

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorAttachment->ID, 0);

        // create a renderbuffer object for depth
        depthAttachment = Texture::create(width, height, TEXTURE_DIFFUSE, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE);

        glBindTexture(GL_TEXTURE_2D, depthAttachment->ID);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);
        glBindTexture(GL_TEXTURE_2D, 0);

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthAttachment->ID, 0);

        // now that we actually created the framebuffer and added all attachments we want to check if it is actually complete now
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            throw std::runtime_error("Framebuffer is not complete!");
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
};

#endif // FRAMEBUFFER_H
