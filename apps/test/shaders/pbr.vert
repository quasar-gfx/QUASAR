#version 410 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;
layout (location = 3) in vec3 aTangent;

out vec2 TexCoords;
out vec3 WorldPos;
out vec3 Normal;
out vec4 fragPos_lS;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
uniform mat3 normalMatrix;
uniform mat4 lightSpaceMatrix;

void main() {
    TexCoords = aTexCoords;
    WorldPos = vec3(model * vec4(aPos, 1.0));
    Normal = normalMatrix * aNormal;

    fragPos_lS  = lightSpaceMatrix * vec4(WorldPos, 1.0);

    gl_Position = projection * view * vec4(WorldPos, 1.0);
}
