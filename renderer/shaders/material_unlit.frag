#include "constants.glsl"
#include "camera.glsl"
#include "depth_peeling.glsl"

layout(location = 0) out vec4 FragColor;
layout(location = 1) out float FragAlpha;
layout(location = 2) out vec4 FragNormal;
layout(location = 3) out uvec4 FragIDs;

in VertexData {
    flat uint DrawID;
    vec2 TexCoord;
    vec3 FragPosView;
    vec3 FragPosWorld;
    vec3 Color;
    vec3 Normal;
    vec3 Tangent;
    vec3 BiTangent;
    vec4 FragPosLightSpace;
} fsIn;

// Material
uniform struct Material {
    vec4 baseColor;
    vec4 baseColorFactor;

    int alphaMode;
    float maskThreshold;

    bool hasBaseColorMap; // use diffuse map

    // Material textures
    sampler2D baseColorMap; // 0
} material;

// Depth peeling helpers moved to depth_peeling.glsl

void main() {
    vec4 baseColor;
    if (material.hasBaseColorMap) {
        baseColor = texture(material.baseColorMap, fsIn.TexCoord) * material.baseColorFactor;
    }
    else {
        baseColor = material.baseColor * material.baseColorFactor;
    }
    baseColor.rgb *= fsIn.Color;

    float alpha = (material.alphaMode == ALPHA_OPAQUE) ? 1.0 : baseColor.a;
    if (alpha < material.maskThreshold)
        discard;

#ifdef DO_DEPTH_PEELING
    applyDepthPeeling(fsIn.FragPosView);
#endif

    FragColor = vec4(baseColor.rgb, alpha);
    FragAlpha = alpha;
    FragNormal = vec4(normalize(fsIn.Normal), 1.0);
    FragIDs = uvec4(fsIn.DrawID, gl_PrimitiveID, 0, material.alphaMode);
    FragIDs.z = floatBitsToUint((-fsIn.FragPosView.z - camera.near) / (camera.far - camera.near));
}
