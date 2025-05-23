#include "tone_map.glsl"

out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D screenColor;
uniform sampler2D screenDepth;
uniform sampler2D screenNormals;
uniform sampler2D screenPositions;
uniform usampler2D idBuffer;

uniform sampler2D ssao;

uniform bool toneMap = true;
uniform float exposure = 1.0;

void main() {
    vec3 color = texture(screenColor, TexCoords).rgb;
    float ambient = texture(ssao, TexCoords).r;
    color *= ambient;

    FragColor = vec4(color, 1.0);
}
