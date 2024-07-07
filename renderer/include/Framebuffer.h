#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H

#include <glad/glad.h>

#include <OpenGLObject.h>
#include <Texture.h>
#include <CubeMap.h>

class Framebuffer : public OpenGLObject {
public:
    explicit Framebuffer() {
        glGenFramebuffers(1, &ID);
    }
    ~Framebuffer() {
        glDeleteFramebuffers(1, &ID);
    }

    bool checkStatus(const std::string &name = "") {
        return glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
    }

    void attachTexture(const Texture &texture, GLenum attachment) {
        if (!texture.multiSampled) {
            glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, texture.ID, 0);
        }
        else {
            glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D_MULTISAMPLE, texture.ID, 0);
        }
    }

    void attachCubeMap(const CubeMap &cubeMap, GLenum attachment) {
        glFramebufferTexture(GL_FRAMEBUFFER, attachment, cubeMap.ID, 0);
    }

    void blitToScreen(unsigned int width, unsigned int height) {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, ID);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    }

    void bind() const {
        glBindFramebuffer(GL_FRAMEBUFFER, ID);
    }

    void unbind() const {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
};

#endif // FRAMEBUFFER_H
