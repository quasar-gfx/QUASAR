#ifndef FRAME_RENDER_TARGET_H
#define FRAME_RENDER_TARGET_H

#include <RenderTargets/RenderTargetBase.h>
#include <RenderTargets/RenderTarget.h>

namespace quasar {

class FrameRenderTarget : public RenderTargetBase {
public:
    Texture colorTexture;
    Texture alphaTexture;
    Texture normalsTexture;
    Texture idTexture;
    Texture depthStencilTexture;

    FrameRenderTarget(const RenderTargetCreateParams& params)
        : RenderTargetBase(params)
        , colorTexture({
            .width = width,
            .height = height,
            .internalFormat = GL_RGBA16F,
            .format = GL_RGBA,
            .type = GL_FLOAT,
            .wrapS = params.wrapS,
            .wrapT = params.wrapT,
            .minFilter = params.minFilter,
            .magFilter = params.magFilter,
            .multiSampled = params.multiSampled,
            .numSamples = params.numSamples,
        })
        , alphaTexture({
            .width = width,
            .height = height,
            .internalFormat = GL_R8,
            .format = GL_RED,
            .type = GL_UNSIGNED_BYTE,
            .wrapS = params.wrapS,
            .wrapT = params.wrapT,
            .minFilter = params.minFilter,
            .magFilter = params.magFilter,
            .multiSampled = params.multiSampled,
            .numSamples = params.numSamples,
        })
        , normalsTexture({
            .width = width,
            .height = height,
            .internalFormat = GL_RGB16F,
            .format = GL_RGB,
            .type = GL_HALF_FLOAT,
            .wrapS = params.wrapS,
            .wrapT = params.wrapT,
            .minFilter = params.minFilter,
            .magFilter = params.magFilter,
            .multiSampled = params.multiSampled,
            .numSamples = params.numSamples,
        })
        , idTexture({
            .width = width,
            .height = height,
            .internalFormat = GL_RGBA32UI,
            .format = GL_RGBA_INTEGER,
            .type = GL_UNSIGNED_INT,
            .wrapS = params.wrapS,
            .wrapT = params.wrapT,
            .minFilter = GL_NEAREST,
            .magFilter = GL_NEAREST,
            .multiSampled = params.multiSampled,
            .numSamples = params.numSamples,
        })
        , depthStencilTexture({
            .width = width,
            .height = height,
            .internalFormat = GL_DEPTH32F_STENCIL8,
            .format = GL_DEPTH_STENCIL,
            .type = GL_FLOAT_32_UNSIGNED_INT_24_8_REV,
            .wrapS = params.wrapS,
            .wrapT = params.wrapT,
            .minFilter = GL_NEAREST,
            .magFilter = GL_NEAREST,
            .multiSampled = params.multiSampled,
            .numSamples = params.numSamples,
        })
    {
        framebuffer.bind();
        framebuffer.attachTexture(colorTexture, GL_COLOR_ATTACHMENT0);
        framebuffer.attachTexture(alphaTexture, GL_COLOR_ATTACHMENT1);
        framebuffer.attachTexture(normalsTexture, GL_COLOR_ATTACHMENT2);
        framebuffer.attachTexture(idTexture, GL_COLOR_ATTACHMENT3);
        framebuffer.attachTexture(depthStencilTexture, GL_DEPTH_STENCIL_ATTACHMENT);

        uint attachments[4] = {
            GL_COLOR_ATTACHMENT0,
            GL_COLOR_ATTACHMENT1,
            GL_COLOR_ATTACHMENT2,
            GL_COLOR_ATTACHMENT3
        };
        glDrawBuffers(4, attachments);

        if (!framebuffer.checkStatus()) {
            throw std::runtime_error("FrameRenderTarget Framebuffer is not complete!");
        }

        framebuffer.unbind();
    }
    ~FrameRenderTarget() = default;

    void blit(RenderTarget& rt, GLenum filter = GL_NEAREST) {
        blit(rt,
             0, 0, width,  height,
             0, 0, rt.width, rt.height,
             filter);
    }

    void blit(RenderTarget& rt,
              GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1,
              GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1,
              GLenum filter = GL_NEAREST)
    {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer.ID);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, rt.getFramebufferID());

        // Color
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        GLenum drawBuffers[] = { GL_COLOR_ATTACHMENT0 };
        glDrawBuffers(1, drawBuffers);
        glBlitFramebuffer(srcX0, srcY0, srcX1, srcY1,
                          dstX0, dstY0, dstX1, dstY1,
                          GL_COLOR_BUFFER_BIT, filter);

        // Depth and stencil
        glBlitFramebuffer(srcX0, srcY0, srcX1, srcY1,
                          dstX0, dstY0, dstX1, dstY1,
                          GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT, GL_NEAREST);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void blit(FrameRenderTarget& frameRT, GLenum filter = GL_NEAREST) {
        blit(frameRT,
             0, 0, width,  height,
             0, 0, frameRT.width, frameRT.height,
             filter);
    }

    void blit(FrameRenderTarget& frameRT,
              GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1,
              GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1,
              GLenum filter = GL_NEAREST)
    {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer.ID);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, frameRT.getFramebufferID());

        // Color
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        GLenum drawBuffers[] = { GL_COLOR_ATTACHMENT0 };
        glDrawBuffers(1, drawBuffers);
        glBlitFramebuffer(srcX0, srcY0, srcX1, srcY1,
                          dstX0, dstY0, dstX1, dstY1,
                          GL_COLOR_BUFFER_BIT, filter);

        // Alpha
        glReadBuffer(GL_COLOR_ATTACHMENT1);
        GLenum drawBuffersA[] = { GL_COLOR_ATTACHMENT1 };
        glDrawBuffers(1, drawBuffersA);
        glBlitFramebuffer(srcX0, srcY0, srcX1, srcY1,
                          dstX0, dstY0, dstX1, dstY1,
                          GL_COLOR_BUFFER_BIT, filter);
        // Normals
        glReadBuffer(GL_COLOR_ATTACHMENT2);
        GLenum drawBuffersN[] = { GL_COLOR_ATTACHMENT2 };
        glDrawBuffers(1, drawBuffersN);
        glBlitFramebuffer(srcX0, srcY0, srcX1, srcY1,
                          dstX0, dstY0, dstX1, dstY1,
                          GL_COLOR_BUFFER_BIT, filter);

        // IDs
        glReadBuffer(GL_COLOR_ATTACHMENT3);
        GLenum drawBuffersID[] = { GL_COLOR_ATTACHMENT3 };
        glDrawBuffers(1, drawBuffersID);
        glBlitFramebuffer(srcX0, srcY0, srcX1, srcY1,
                          dstX0, dstY0, dstX1, dstY1,
                          GL_COLOR_BUFFER_BIT, GL_NEAREST);

        // Depth and stencil
        glBlitFramebuffer(srcX0, srcY0, srcX1, srcY1,
                          dstX0, dstY0, dstX1, dstY1,
                          GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT, GL_NEAREST);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void blitColorAndDepth(FrameRenderTarget& frameRT, GLenum filter = GL_NEAREST) {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer.ID);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, frameRT.getFramebufferID());

        // Color
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        GLenum drawBuffers[] = { GL_COLOR_ATTACHMENT0 };
        glDrawBuffers(1, drawBuffers);
        glBlitFramebuffer(0, 0, width, height,
                          0, 0, frameRT.width, frameRT.height,
                          GL_COLOR_BUFFER_BIT, filter);

        // Depth and stencil
        glBlitFramebuffer(0, 0, width, height,
                          0, 0, frameRT.width, frameRT.height,
                          GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT, GL_NEAREST);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void blitToScreen(uint width, uint height) {
        framebuffer.blitToScreen(width, height);
    }

    void resize(uint width, uint height) override {
        RenderTargetBase::resize(width, height);

        colorTexture.resize(width, height);
        alphaTexture.resize(width, height);
        normalsTexture.resize(width, height);
        idTexture.resize(width, height);
        depthStencilTexture.resize(width, height);
    }

    void readPixels(unsigned char* data, bool readAsFloat = false) {
        bind();
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        colorTexture.readPixels(data, readAsFloat);
        unbind();
    }

    void writeColorAsPNG(const std::string& path) {
        bind();
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        colorTexture.writeToPNG(path);
        unbind();
    }

    void writeAlphaAsPNG(const std::string& path) {
        bind();
        glReadBuffer(GL_COLOR_ATTACHMENT1);
        alphaTexture.writeToPNG(path);
        unbind();
    }

    void writeColorAsJPG(const std::string& path, int quality = 85) {
        bind();
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        colorTexture.writeToJPG(path, quality);
        unbind();
    }

    void writeColorAsHDR(const std::string& path) {
        bind();
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        colorTexture.writeToHDR(path);
        unbind();
    }

    void writeColorJPGToMemory(std::vector<unsigned char>& outputData, int quality = 85) {
        bind();
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        colorTexture.writeJPGToMemory(outputData, quality);
        unbind();
    }

    void writeAlphaToMemory(std::vector<unsigned char>& outputData) {
        bind();
        glReadBuffer(GL_COLOR_ATTACHMENT1);
        alphaTexture.readPixels(outputData.data(), false);
        unbind();
    }
};

} // namespace quasar

#endif // FRAME_RENDER_TARGET_H
