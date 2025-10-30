#include "constants.glsl"
#include "tonemap.glsl"

out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D screenColor;
uniform sampler2D screenDepth;
uniform sampler2D screenNormals;
uniform sampler2D screenPositions;
uniform usampler2D idTexture;

uniform bool tonemap = true;
uniform float exposure = 1.0;

uniform float depthThreshold;

uniform int searchRadius = 3;

void main() {
    vec3 color = texture(screenColor, TexCoord).rgb;
    float centerDepth = texture(screenDepth, TexCoord).r;

    if (centerDepth == 0.0 || centerDepth >= MAX_DEPTH) {
        bool isSkyBox = (texture(screenNormals, TexCoord).xyz == vec3(0.0/0.0) &&
                         texture(idTexture, TexCoord).z == 0xFFFFFFFFu);

        // Fill hole
        vec2 textureSize = vec2(textureSize(screenColor, 0));
        if (!isSkyBox) {
            vec3 sumColor = vec3(0.0);
            float sumWeight = 0.0;

            for (int x = -searchRadius; x <= searchRadius; x++) {
                for (int y = -searchRadius; y <= searchRadius; y++) {
                    vec2 texCoord = TexCoord + vec2(x, y) / textureSize;
                    float sampleDepth = texture(screenDepth, texCoord).r;
                    if (sampleDepth < MAX_DEPTH) {
                        float weight = 1.0 / (1.0 + abs(centerDepth - sampleDepth));
                        sumColor  += texture(screenColor, texCoord).rgb * weight;
                        sumWeight += weight;
                    }
                }
            }

            if (sumWeight > 0.0) {
                color = sumColor / sumWeight;
            }
        }
    }

    if (tonemap) {
        color = tonemapExponential(color, exposure);
    }

    FragColor = vec4(color, 1.0);
}
