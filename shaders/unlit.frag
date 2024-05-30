#version 410 core
layout(location = 0) out vec4 positionBuffer;
layout(location = 1) out vec4 normalsBuffer;
layout(location = 2) out vec4 idBuffer;
layout(location = 3) out vec4 FragColor;

in VertexData {
    flat uint VertexID;
    vec2 TexCoords;
    vec3 FragPos;
    vec3 Color;
    vec3 Normal;
    vec3 Tangent;
    vec3 BiTangent;
    vec4 FragPosLightSpace;
} fsIn;

// material textures
uniform sampler2D diffuseMap; // 0

uniform vec3 overrideColor;
uniform bool transparent;

uniform vec3 camPos;

void main() {
    vec4 color = texture(diffuseMap, fsIn.TexCoords);
    float alpha = (transparent) ? color.a : 1.0;
    if (alpha < 0.1)
        discard;

    if (overrideColor != vec3(0.0)) {
        color.rgb = overrideColor;
    }
    else {
        color.rgb *= fsIn.Color;
    }

    vec3 norm = normalize(fsIn.Normal);
    vec3 viewDir = normalize(camPos - fsIn.FragPos);

    positionBuffer = vec4(fsIn.FragPos, 1.0);
    normalsBuffer = vec4(normalize(fsIn.Normal), 1.0);
    idBuffer = vec4(fsIn.VertexID, 0.0, 0.0, 0.0);
    FragColor = color;
}
