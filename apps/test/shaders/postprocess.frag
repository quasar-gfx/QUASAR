#version 410 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D screenColor;
uniform sampler2D screenDepth;

void main() {
    vec2 uv = TexCoords;

    vec3 col = texture(screenColor, uv).rgb;
    FragColor = vec4(col, 1.0);
}
