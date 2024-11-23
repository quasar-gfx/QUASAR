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

#endif // UTILS_H
