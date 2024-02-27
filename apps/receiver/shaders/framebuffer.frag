#version 330 core
out vec4 FragColor;

in vec2 normal;
in vec2 texCoords;
in vec2 tangent;

uniform sampler2D texture_diffuse1;

void main() {
    vec4 col = texture(texture_diffuse1, texCoords);
    if (col.a < 0.5) {
        discard;
    }

    FragColor = col;
}
