#version 330 core
out vec4 FragColor;

in vec3 TexCoords;

uniform samplerCube skybox;

void main() {
    vec4 col = texture(skybox, TexCoords);
    FragColor = col;
}
