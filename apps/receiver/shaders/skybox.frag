#version 330 core
out vec4 FragColor;

in vec3 TexCoords;

uniform samplerCube environmentMap;

void main() {
    vec4 col = texture(environmentMap, TexCoords);
    FragColor = col;
}
