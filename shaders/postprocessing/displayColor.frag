out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D screenColor;
uniform sampler2D screenDepth;
uniform sampler2D screenPositions;
uniform sampler2D screenNormals;
uniform sampler2D idBuffer;

uniform int maxLayers;
uniform sampler2D peelingLayers[2];

uniform bool doToneMapping = true;
uniform float exposure = 1.0;

void main() {
    vec4 color = vec4(0.0);
    for (int i = 0; i < maxLayers; i++) {
        vec4 layerColor = texture(peelingLayers[i], TexCoords);
        color.rgb += layerColor.rgb * (1.0 - color.a);
        color.a += layerColor.a * (1.0 - color.a);
    }

    if (doToneMapping) {
        vec3 hdrCol = color.rgb;
        vec3 toneMappedResult = vec3(1.0) - exp(-hdrCol * exposure);
        FragColor = vec4(toneMappedResult, 1.0);
    }
    else {
        FragColor = color;
    }
}
