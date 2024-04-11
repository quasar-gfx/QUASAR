#version 410 core
layout(location = 0) out vec4 positionBuffer;
layout(location = 1) out vec4 normalsBuffer;
layout(location = 2) out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoords;

// material parameters
uniform sampler2D albedoMap;
uniform sampler2D specularMap;
uniform sampler2D normalMap;
uniform sampler2D metallicMap;
uniform sampler2D roughnessMap;
uniform sampler2D aoMap;
uniform float shininess;

uniform vec3 camPos;

uniform samplerCube environmentMap;

void main() {
    vec4 col = texture(albedoMap, TexCoords);
    float alpha = col.a;
    if (alpha < 0.1)
        discard;

    vec3 norm = normalize(Normal);
    vec3 viewDir = normalize(camPos - FragPos);

    positionBuffer = vec4(FragPos, 1.0);
    normalsBuffer = vec4(normalize(Normal), 1.0);
    FragColor = col;
}
