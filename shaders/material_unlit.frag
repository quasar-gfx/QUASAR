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

// material
struct Material {
    vec3 baseColor;
    float opacity;
    bool transparent;

    // material textures
    sampler2D diffuseMap; // 0
};

uniform Material material;

uniform vec3 camPos;

void main() {
    vec4 color = texture(material.diffuseMap, fsIn.TexCoords);
    if (color.rgb == vec3(0.0) && material.baseColor != vec3(-1.0)) {
        color.rgb = material.baseColor;
        color.a = material.opacity;
    }
    else {
        color.rgb *= fsIn.Color;
    }

    float alpha = (material.transparent) ? color.a : 1.0;
    if (alpha < 0.1)
        discard;

    vec3 norm = normalize(fsIn.Normal);
    vec3 viewDir = normalize(camPos - fsIn.FragPos);

    positionBuffer = vec4(fsIn.FragPos, 1.0);
    normalsBuffer = vec4(normalize(fsIn.Normal), 1.0);
    idBuffer = vec4(fsIn.VertexID, 0.0, 0.0, 0.0);
    FragColor = vec4(color.rgb, alpha);
}
