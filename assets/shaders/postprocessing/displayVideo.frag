#version 410 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D screenPositions;
uniform sampler2D screenNormals;
uniform sampler2D screenColor;
uniform sampler2D screenDepth;
uniform sampler2D videoTexture;

void main() {
    vec3 col;
    if (TexCoords.x < 0.05 || TexCoords.x > 0.95 || TexCoords.y < 0.05 || TexCoords.y > 0.95) {
        col = texture(screenColor, TexCoords).rgb;
    }
    else {
        col = texture(videoTexture, TexCoords).rgb;
    }
    FragColor = vec4(col, 1.0);
}
