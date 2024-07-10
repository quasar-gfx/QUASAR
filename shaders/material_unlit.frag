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

const int AlphaOpaque      = 0;
const int AlphaMasked      = 1;
const int AlphaTransparent = 2;

// material
struct Material {
    vec3 baseColor;
    float opacity;

    int alphaMode;
    float maskThreshold;

    bool diffuseMapped; // use diffuse map

    // material textures
    sampler2D diffuseMap; // 0
};

uniform Material material;

uniform vec3 camPos;

void main() {
    // material properties
    vec4 color = texture(material.diffuseMap, fsIn.TexCoords);

    if (!material.diffuseMapped) {
        color.rgb = material.baseColor;
        color.a = material.opacity;
    }
    else {
        color.rgb *= fsIn.Color;
    }

    float alpha = (material.alphaMode == AlphaOpaque) ? 1.0 : color.a;
    if (alpha < material.maskThreshold)
        discard;

    positionBuffer = vec4(fsIn.FragPos, 1.0);
    normalsBuffer = vec4(normalize(fsIn.Normal), 1.0);
    idBuffer = vec4(fsIn.VertexID, 0.0, 0.0, 0.0);
    FragColor = vec4(color.rgb, alpha);
}
