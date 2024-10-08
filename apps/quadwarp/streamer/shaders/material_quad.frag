layout(location = 0) out vec4 FragColor;
layout(location = 1) out vec4 FragPosition;
layout(location = 2) out vec4 FragNormal;
layout(location = 3) out vec4 FragIDs;

in VertexData {
    flat uint VertexID;
    noperspective vec4 TexCoords;
    vec3 FragPos;
    vec3 Color;
    vec3 Normal;
    vec3 Tangent;
    vec3 BiTangent;
    vec4 FragPosLightSpace;
} fsIn;

// material
struct Material {
    vec4 baseColor;
    vec4 baseColorFactor;

    int alphaMode;
    float maskThreshold;

    bool hasBaseColorMap; // use diffuse map

    // material textures
    sampler2D baseColorMap; // 0
};

uniform Material material;

uniform vec3 camPos;

void main() {
    vec4 baseColor;
    if (material.hasBaseColorMap) {
        baseColor = texture(material.baseColorMap, fsIn.TexCoords.xy / fsIn.TexCoords.w) * material.baseColorFactor;
    }
    else {
        baseColor = material.baseColor * material.baseColorFactor;
    }
    baseColor.rgb *= fsIn.Color;

    float alpha = (material.alphaMode == ALPHA_OPAQUE) ? 1.0 : baseColor.a;
    if (alpha < material.maskThreshold)
        discard;

    FragColor = vec4(baseColor.rgb, alpha);
    FragPosition = vec4(fsIn.FragPos, 1.0);
    FragNormal = vec4(normalize(fsIn.Normal), 1.0);
    FragIDs = vec4(gl_PrimitiveID, fsIn.VertexID, 0.0, 0.0);
}
