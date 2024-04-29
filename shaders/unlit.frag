#version 410 core
layout(location = 0) out vec4 positionBuffer;
layout(location = 1) out vec4 normalsBuffer;
layout(location = 2) out vec4 FragColor;

in vec2 TexCoords;
in vec3 FragPos;
in vec3 Normal;
in vec3 Tangent;
in vec3 BiTangent;
in vec2 TexCoords;
out vec4 FragPosLightSpace;

// material parameters
uniform sampler2D diffuseMap;
uniform sampler2D specularMap;
uniform float shininess;

uniform vec3 camPos;

uniform samplerCube environmentMap;

void main() {
    vec4 col = texture(diffuseMap, TexCoords);
    float alpha = col.a;
    if (alpha < 0.5)
        discard;

    vec3 norm = normalize(Normal);
    vec3 viewDir = normalize(camPos - FragPos);

    positionBuffer = vec4(FragPos, 1.0);
    normalsBuffer = vec4(normalize(Normal), 1.0);
    FragColor = col;
}
