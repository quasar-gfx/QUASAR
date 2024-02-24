#version 330 core
out vec4 FragColor;

in vec2 normal;
in vec2 texCoords;
in vec2 tangent;

uniform sampler2D texture_diffuse1;

void main() {
    FragColor = texture(texture_diffuse1, texCoords);
}
