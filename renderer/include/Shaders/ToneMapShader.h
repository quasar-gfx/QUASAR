#ifndef TONE_MAP_SHADER_H
#define TONE_MAP_SHADER_H

#include <Shaders/Shader.h>

class ToneMapShader : public Shader {
public:
    ToneMapShader()
            : Shader({
                .vertexCodeData = SHADER_BUILTIN_POSTPROCESS_VERT,
                .vertexCodeSize = SHADER_BUILTIN_POSTPROCESS_VERT_len,
                .fragmentCodeData = SHADER_BUILTIN_TONEMAP_FRAG,
                .fragmentCodeSize = SHADER_BUILTIN_TONEMAP_FRAG_len
            }) {}
};

#endif // TONE_MAP_SHADER_H
