#include "constants.glsl"
#include "toneMap.glsl"

out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D screenColor;
uniform sampler2D screenDepth;
uniform sampler2D screenNormals;
uniform usampler2D idBuffer;

uniform bool toneMap = true;
uniform float exposure = 1.0;

uniform float depthThreshold;

uniform int searchRadius = 3;

void main() {
    vec3 color = texture(screenColor, TexCoords).rgb;
    float centerDepth = texture(screenDepth, TexCoords).r;

    if (centerDepth >= MAX_DEPTH) {
        vec2 textureSize = vec2(textureSize(screenColor, 0));

        bool isSkyBox = true;
        for (int i = 1; i <= searchRadius; i++) {
            float topDepth = texture(screenDepth, TexCoords + vec2(0.0, i / textureSize.y)).r;
            float bottomDepth = texture(screenDepth, TexCoords - vec2(0.0, i / textureSize.y)).r;
            float leftDepth = texture(screenDepth, TexCoords - vec2(i / textureSize.x, 0.0)).r;
            float rightDepth = texture(screenDepth, TexCoords + vec2(i / textureSize.x, 0.0)).r;

            bool bothSidesUnder =
                ((abs(topDepth - bottomDepth) <= depthThreshold) &&
                 (abs(topDepth - centerDepth) > depthThreshold) &&
                 (abs(bottomDepth - centerDepth) > depthThreshold)) ||
                ((abs(leftDepth - rightDepth) <= depthThreshold) &&
                 (abs(leftDepth - centerDepth) > depthThreshold) &&
                 (abs(rightDepth - centerDepth) > depthThreshold));
            bothSidesUnder = bothSidesUnder ||
                             ((topDepth < MAX_DEPTH && bottomDepth < MAX_DEPTH) ||
                              (leftDepth < MAX_DEPTH && rightDepth < MAX_DEPTH));
            if (bothSidesUnder) {
                isSkyBox = false;
                break;
            }
        }

        // fill hole
        if (!isSkyBox) {
            vec3 sumColor = vec3(0.0);
            float sumWeight = 0.0;

            for (int x = -searchRadius; x <= searchRadius; x++) {
                for (int y = -searchRadius; y <= searchRadius; y++) {
                    vec2 texCoords = TexCoords + vec2(x, y) / textureSize;
                    float sampleDepth = texture(screenDepth, texCoords).r;
                    if (sampleDepth < MAX_DEPTH) {
                        float weight = 1.0 / (1.0 + abs(centerDepth - sampleDepth));
                        sumColor += texture(screenColor, texCoords).rgb * weight;
                        sumWeight += weight;
                    }
                }
            }

            if (sumWeight > 0.0) {
                color = sumColor / sumWeight;
            }
        }
    }

    if (toneMap) {
        color = applyToneMapExponential(color, exposure);
        color = linearToSRGB(color);
    }

    FragColor = vec4(color, 1.0);
}
