#version 330 core
out vec4 FragColor;

in vec3 TexCoords;

struct AmbientLight {
    vec3 color;
    float intensity;
};

uniform AmbientLight ambientLight;

uniform samplerCube skybox;

void main() {
    vec4 col = texture(skybox, TexCoords);
    col.rgb = ambientLight.intensity * (ambientLight.color * col.rgb);
    FragColor = col;
}
