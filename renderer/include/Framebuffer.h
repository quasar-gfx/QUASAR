#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H

#include <OpenGLObject.h>
#include <Texture.h>
#include <CubeMap.h>

namespace quasar {

class Framebuffer : public OpenGLObject {
public:
    unsigned int numAttachments = 0;

    Framebuffer() {
        glGenFramebuffers(1, &ID);
    }
    ~Framebuffer() override {
        glDeleteFramebuffers(1, &ID);
    }

    bool checkStatus(const std::string &name = "") {
        GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (status != GL_FRAMEBUFFER_COMPLETE) {
            std::cerr << "Framebuffer is not complete: ";
            switch (status) {
            case GL_FRAMEBUFFER_UNDEFINED:
                std::cerr << "GL_FRAMEBUFFER_UNDEFINED";
                break;
            case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
                std::cerr << "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT";
                break;
            case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
                std::cerr << "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT";
                break;
#ifdef GL_CORE
            case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
                std::cerr << "GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER";
                break;
            case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
                std::cerr << "GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER";
                break;
#endif
            case GL_FRAMEBUFFER_UNSUPPORTED:
                std::cerr << "GL_FRAMEBUFFER_UNSUPPORTED";
                break;
            case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
                std::cerr << "GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE";
                break;
            case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
                std::cerr << "GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS";
                break;
            default:
                std::cerr << "Unknown error";
            }
            std::cerr << std::endl;
            return false;
        }
        return true;
    }

    void attachTexture(const Texture &texture, GLenum attachment) {
        if (!texture.multiSampled) {
            glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, texture.ID, 0);
        }
        else {
            glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D_MULTISAMPLE, texture.ID, 0);
        }
        numAttachments++;
    }

    void attachCubeMap(const CubeMap &cubeMap, GLenum attachment) {
        glFramebufferTexture(GL_FRAMEBUFFER, attachment, cubeMap.ID, 0);
        numAttachments++;
    }

    void blitToScreen(unsigned int width, unsigned int height) {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, ID);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    }

    void bind() const override {
        glBindFramebuffer(GL_FRAMEBUFFER, ID);
    }

    void unbind() const override {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
};

} // namespace quasar

#endif // FRAMEBUFFER_H
