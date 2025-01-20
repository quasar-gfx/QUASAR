out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D screenColor;
uniform sampler2D screenDepth;
uniform sampler2D screenPositions;
uniform sampler2D screenNormals;
uniform usampler2D idBuffer;

#define MAX_DEPTH 0.9999

uniform bool toneMap = true;
uniform bool gammaCorrect = true;
uniform float exposure = 1.0;

uniform int searchRadius = 2;

vec3 linearToSRGB(vec3 color) {
    return mix(pow(color, vec3(1.0 / 2.4)) * 1.055 - 0.055, color * 12.92, lessThanEqual(color, vec3(0.0031308)));
}

void main() {
    vec3 color = texture(screenColor, TexCoords).rgb;
    vec2 textureSize = vec2(textureSize(screenColor, 0));

    float depth = texture(screenDepth, TexCoords).r;
    if (depth >= MAX_DEPTH) {
        // check if opposite ends both are under threshold
        float topDepth    = texture(screenDepth, TexCoords + vec2(0.0, -searchRadius) / textureSize).r;
        float bottomDepth = texture(screenDepth, TexCoords + vec2(0.0, searchRadius) / textureSize).r;
        float leftDepth   = texture(screenDepth, TexCoords + vec2(-searchRadius, 0.0) / textureSize).r;
        float rightDepth  = texture(screenDepth, TexCoords + vec2(searchRadius, 0.0) / textureSize).r;
        bool bothSidesUnder = (topDepth < MAX_DEPTH && bottomDepth < MAX_DEPTH) ||
                              (leftDepth < MAX_DEPTH && rightDepth < MAX_DEPTH);

        if (bothSidesUnder) {
            // average the surrounding pixels whose depth is less than MAX_DEPTH
            vec3 sum = vec3(0.0);
            int count = 0;
            for (int x = -searchRadius; x <= searchRadius; x++) {
                for (int y = -searchRadius; y <= searchRadius; y++) {
                    vec2 offset = vec2(x, y) / textureSize;
                    float sampleDepth = texture(screenDepth, TexCoords + offset).r;
                    if (sampleDepth < MAX_DEPTH) {
                        sum += texture(screenColor, TexCoords + offset).rgb;
                        count++;
                    }
                }
            }
            if (count > 0) color = sum / float(count);
        }
    }

    if (toneMap) {
        vec3 toneMappedResult = vec3(1.0) - exp(-color.rgb * exposure);
        color = toneMappedResult;
        if (gammaCorrect) {
            color = linearToSRGB(color);
        }
    }

    FragColor = vec4(color, 1.0);
}
