layout(location = 0) out vec4 FragColor;
layout(location = 1) out float FragAlpha;
layout(location = 2) out vec4 FragNormal;
layout(location = 3) out uvec4 FragIDs;

in VertexData {
    flat uint DrawID;
    vec3 TexCoord3D;
    vec3 FragPos;
} fsIn;

// Material
uniform struct Material {
    vec4 baseColor;
    vec4 baseColorFactor;

    int alphaMode;

    bool hasBaseColorMap; // use color map
    bool hasAlphaMap; // use alpha map

    // Material textures
    sampler2D baseColorMap; // 0
    sampler2D alphaMap; // 1
} material;

void main() {
    vec4 color;
    vec2 uv = fsIn.TexCoord3D.xy / fsIn.TexCoord3D.z;
    if (material.hasBaseColorMap) {
        color = texture(material.baseColorMap, uv) * material.baseColorFactor;
    }
    else {
        color = material.baseColor * material.baseColorFactor;
    }

    float alpha = 1.0;
    if (material.alphaMode == ALPHA_BLEND && material.hasAlphaMap) {
        alpha = texture(material.alphaMap, uv).r;
        if (alpha == 0.0) { // HACK: This is most likely an expanded edge, so show anyways
            alpha = 1.0;
        }
    }

    FragColor = vec4(color.rgb, alpha);
    // FragNormal = vec4(normalize(fsIn.Normal), 1.0);
    FragIDs = uvec4(fsIn.DrawID, gl_PrimitiveID, 0.0, 1.0);
}
