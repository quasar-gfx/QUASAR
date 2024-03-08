#version 410 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D screenColor;
uniform sampler2D screenDepth;
uniform sampler2D videoTexture;

void main() {
    vec2 uv = TexCoords;

    vec3 col;
    if (uv.x < 0.05 || uv.x > 0.95 || uv.y < 0.05 || uv.y > 0.95) {
        col = texture(screenColor, uv).rgb;
    }
    else {
        col = texture(videoTexture, uv).rgb;
    }
    FragColor = vec4(col, 1.0);
}
