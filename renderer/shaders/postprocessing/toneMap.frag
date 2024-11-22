out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D screenColor;
uniform sampler2D screenDepth;
uniform sampler2D screenPositions;
uniform sampler2D screenNormals;
uniform usampler2D idBuffer;

uniform bool toneMap = true;
uniform bool gammaCorrect = false;
uniform float exposure = 1.0;

vec3 linearToSRGB(vec3 color) {
    return mix(pow(color, vec3(1.0 / 2.4)) * 1.055 - 0.055, color * 12.92, lessThanEqual(color, vec3(0.0031308)));
}

void main() {
    vec3 color = texture(screenColor, TexCoords).rgb;
    if (toneMap) {
        vec3 toneMappedResult = vec3(1.0) - exp(-color.rgb * exposure);
        color = toneMappedResult;
    }
    if (gammaCorrect) {
        color = linearToSRGB(color);
    }
    FragColor = vec4(color, 1.0);
}
