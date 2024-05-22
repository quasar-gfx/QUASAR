#version 410 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D screenPositions;
uniform sampler2D screenNormals;
uniform sampler2D idBuffer;
uniform sampler2D screenColor;
uniform sampler2D screenDepth;

uniform float exposure = 1.0;

void main() {
    vec3 hdrCol = texture(screenColor, TexCoords).rgb;
    vec3 toneMappedResult = vec3(1.0) - exp(-hdrCol * exposure);
    FragColor = vec4(toneMappedResult, 1.0);
}
