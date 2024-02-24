#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aNormal;
layout (location = 2) in vec2 aTexCoords;
layout (location = 3) in vec2 aTangent;

out vec2 normal;
out vec2 texCoords;
out vec2 tangent;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    normal = aNormal;
    texCoords = aTexCoords;
    tangent = aTangent;

    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
