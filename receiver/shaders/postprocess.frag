#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D screenTexture;
uniform sampler2D videoTexture;

void main() {
    vec2 uv = TexCoords;

    vec3 col;
    if (uv.x < 0.5) {
        col = texture(screenTexture, uv).rgb;
    }
    else {
        uv.y = 1.0 - uv.y;
        col = texture(videoTexture, uv).rgb;
    }
    FragColor = vec4(col, 1.0);
}
