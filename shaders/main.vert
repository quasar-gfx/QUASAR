#version 410 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;
layout (location = 3) in vec3 aTangent;
layout (location = 4) in vec3 aBitangent;

out vec2 TexCoords;
out vec3 FragPos;
out vec3 Normal;
out vec3 Tangent;
out vec3 BiTangent;
out vec4 FragPosLightSpace;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
uniform mat3 normalMatrix;
uniform mat4 lightSpaceMatrix;

void main() {
    TexCoords = aTexCoords;

    FragPos = vec3(model * vec4(aPos, 1.0));

    Normal = normalMatrix * aNormal;

    Tangent = normalize(vec3(model * vec4(aTangent, 1.0)));
    BiTangent = normalize(vec3(model * vec4(aBitangent, 1.0)));

    FragPosLightSpace = lightSpaceMatrix * vec4(FragPos, 1.0);

    gl_PointSize = 5.0;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
