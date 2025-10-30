#include "tonemap.glsl"

layout(location = 0) out vec4 FragColor;

in vec2 TexCoord;

#ifdef ANDROID
flat in float IsLeftEye;

uniform sampler2DArray screenColor;
uniform sampler2DArray screenDepth;
uniform sampler2DArray screenNormals;
uniform sampler2DArray screenPositions;
uniform usampler2DArray idTexture;
#else
uniform sampler2D screenColor;
uniform sampler2D screenDepth;
uniform sampler2D screenNormals;
uniform sampler2D screenPositions;
uniform usampler2D idTexture;
#endif

uniform bool tonemap;
uniform float exposure;

void main() {
#ifdef ANDROID
    uint viewIdx = gl_ViewID_OVR;
    vec3 color = texture(screenColor, vec3(TexCoord, float(viewIdx))).rgb;
#else
    vec3 color = texture(screenColor, TexCoord).rgb;
#endif

    if (tonemap) {
        color = tonemapExponential(color, exposure);
    }
    FragColor = vec4(color, 1.0);
}
