layout(location = 0) out vec4 FragColor;
layout(location = 1) out vec4 FragNormal;
layout(location = 2) out uvec4 FragIDs;

in VertexData {
    flat unsigned int drawID;
    vec3 TexCoords3D;
    vec3 FragPos;
} fsIn;

// material
uniform struct Material {
    vec4 baseColor;
    vec4 baseColorFactor;

    int alphaMode;
    float maskThreshold;

    bool hasBaseColorMap; // use diffuse map

    // material textures
    sampler2D baseColorMap; // 0
} material;

void main() {
    vec4 baseColor;
    if (material.hasBaseColorMap) {
        vec2 uv = fsIn.TexCoords3D.xy / fsIn.TexCoords3D.z;
        baseColor = texture(material.baseColorMap, uv) * material.baseColorFactor;
    }
    else {
        baseColor = material.baseColor * material.baseColorFactor;
    }

    float alpha = (material.alphaMode == ALPHA_OPAQUE) ? 1.0 : baseColor.a;
    if (alpha < material.maskThreshold)
        discard;

    FragColor = vec4(baseColor.rgb, alpha);
    // FragNormal = vec4(normalize(fsIn.Normal), 1.0);
    FragIDs = uvec4(fsIn.drawID, gl_PrimitiveID, 0.0, 1.0);
}
