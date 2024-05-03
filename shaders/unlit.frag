#version 410 core
layout(location = 0) out vec4 positionBuffer;
layout(location = 1) out vec4 normalsBuffer;
layout(location = 2) out vec4 FragColor;

in vec2 TexCoords;
in vec3 FragPos;
in vec3 Normal;
in vec3 Tangent;
in vec3 BiTangent;
in vec4 FragPosLightSpace;

// material textures
uniform sampler2D diffuseMap; // 0
uniform sampler2D specularMap; // 1
uniform float shininess;

uniform bool transparent;

uniform vec3 camPos;

void main() {
    vec4 color = texture(diffuseMap, TexCoords);
    float alpha = (transparent) ? color.a : 1.0;
    if (alpha < 0.1)
        discard;

    vec3 norm = normalize(Normal);
    vec3 viewDir = normalize(camPos - FragPos);

    positionBuffer = vec4(FragPos, 1.0);
    normalsBuffer = vec4(normalize(Normal), 1.0);
    FragColor = color;
}
