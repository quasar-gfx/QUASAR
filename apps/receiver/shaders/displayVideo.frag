#version 410 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D screenPositions;
uniform sampler2D screenNormals;
uniform sampler2D screenColor;
uniform sampler2D screenDepth;

uniform sampler2D videoTexture;

void main() {
    FragColor = vec4(texture(videoTexture, TexCoords).rgb, 1.0);
}
