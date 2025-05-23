#ifndef TONE_MAP_SHADER_H
#define TONE_MAP_SHADER_H

#include <Shaders/Shader.h>

namespace quasar {

class ToneMapShader : public Shader {
public:
    ToneMapShader()
            : Shader({
                .vertexCodeData = SHADER_BUILTIN_POSTPROCESS_VERT,
                .vertexCodeSize = SHADER_BUILTIN_POSTPROCESS_VERT_len,
                .fragmentCodeData = SHADER_BUILTIN_TONE_MAP_FRAG,
                .fragmentCodeSize = SHADER_BUILTIN_TONE_MAP_FRAG_len
            }) {}
};

} // namespace quasar

#endif // TONE_MAP_SHADER_H
