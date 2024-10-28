#ifndef UTILS_H
#define UTILS_H

#include <iomanip>

#include <Shaders/Shader.h>
#include <RenderTargets/RenderTargetBase.h>

std::string to_string_with_precision(float value, int sig_figs = 3) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(sig_figs - 1) << value;
    return ss.str();
}

void saveRenderTargetToFile(OpenGLRenderer &renderer, const Shader &shader, const std::string &filename, const glm::uvec2 &size, bool saveAsHDR = false) {
    RenderTarget renderTargetTemp({
        .width = size.x,
        .height = size.y,
        .internalFormat = GL_RGBA,
        .format = GL_RGBA,
        .type = GL_UNSIGNED_BYTE,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR
    });

    shader.bind();
    shader.setBool("gammaCorrect", true);
    renderer.drawToRenderTarget(shader, renderTargetTemp);
    shader.setBool("gammaCorrect", false);

    if (saveAsHDR) {
        renderTargetTemp.saveColorAsHDR(filename + ".hdr");
    }
    else {
        renderTargetTemp.saveColorAsPNG(filename + ".png");
    }
}

#endif // UTILS_H
