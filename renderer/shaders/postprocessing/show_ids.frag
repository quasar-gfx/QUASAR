layout(location = 0) out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D screenColor;
uniform sampler2D screenDepth;
uniform sampler2D screenNormals;
uniform sampler2D screenPositions;
uniform usampler2D idTexture;

uniform bool showObjectIDs = true;

float hash(uint x) {
    x = ((x >> 16u) ^ x) * 0x45d9f3bu;
    x = ((x >> 16u) ^ x) * 0x45d9f3bu;
    x = (x >> 16u) ^ x;
    return float(x % 1000u) / 1000.0;
}

vec3 randomColor(uint id) {
    return vec3(hash(id), hash(id + 1u), hash(id + 2u));
}

void main() {
    uvec3 ids = texture(idTexture, TexCoord).rgb;
    uint id = showObjectIDs ? ids.r : ids.g;

    vec3 col = randomColor(id);
    FragColor = vec4(col, 1.0);
}
