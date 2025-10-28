#include "constants.glsl"
#include "camera.glsl"
#include "depth_peeling.glsl"

layout(location = 0) out vec3 gAlbedo;
layout(location = 1) out float gAlpha;
layout(location = 2) out vec3 gPBR;
layout(location = 3) out vec4 gEmissive;
layout(location = 4) out vec3 gNormal;
layout(location = 5) out vec3 gPosition;
layout(location = 6) out vec4 gLightPosition;
layout(location = 7) out uvec4 gIDs;

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

    vec3 emissiveFactor;

    float metallic;
    float metallicFactor;
    float roughness;
    float roughnessFactor;

    bool hasBaseColorMap; // use albedo map
    bool hasNormalMap; // use normal map
    bool hasMetallicMap; // use metallic map
    bool hasRoughnessMap; // use roughness map
    bool hasAOMap; // use ao map
    bool hasEmissiveMap; // use emissive map
    bool metalRoughnessCombined; // use combined metal/roughness map

    // Material textures
    sampler2D baseColorMap; // 0
    sampler2D normalMap; // 1
    sampler2D metallicMap; // 2
    sampler2D roughnessMap; // 3
    sampler2D aoMap; // 4
    sampler2D emissiveMap; // 5

    // IBL
    float IBL; // IBL contribution
    samplerCube irradianceMap; // 6
    samplerCube prefilterMap; // 7
    sampler2D brdfLUT; // 8
} material;


vec3 getNormal() {
	vec3 N = normalize(fsIn.Normal);
	vec3 T = normalize(fsIn.Tangent);
	vec3 B = normalize(fsIn.BiTangent);

    if (!material.hasNormalMap)
        return N;

    if (any(isnan(B))) {
        vec3 q1 = dFdx(fsIn.FragPosWorld);
        vec3 q2 = dFdy(fsIn.FragPosWorld);
        vec2 st1 = dFdx(fsIn.TexCoord);
        vec2 st2 = dFdy(fsIn.TexCoord);

        T = normalize(q1 * st2.t - q2 * st1.t);
        B = -normalize(cross(N, T));
    }

	mat3 TBN = mat3(T, B, N);
	vec3 tangentNormal = texture(material.normalMap, fsIn.TexCoord).xyz * 2.0 - 1.0;
	return normalize(TBN * tangentNormal);
}

void main() {
    vec4 baseColor;
    if (material.hasBaseColorMap) {
        baseColor = texture(material.baseColorMap, fsIn.TexCoord) * material.baseColorFactor;
    }
    else {
        baseColor = material.baseColorFactor;
    }
    baseColor.rgb *= fsIn.Color;

    // Albedo
    vec3 albedo = baseColor.rgb;
    float alpha = (material.alphaMode == ALPHA_OPAQUE) ? 1.0 : baseColor.a;
    if (alpha < material.maskThreshold)
        discard;

#ifdef DO_DEPTH_PEELING
    applyDepthPeeling(fsIn.FragPosView);
#endif

    // Metallic and roughness properties
    float metallic, roughness;
    if (material.metalRoughnessCombined) {
        vec4 mr = texture(material.metallicMap, fsIn.TexCoord);
        metallic = (!material.hasMetallicMap) ? material.metallic : mr.b;
        roughness = (!material.hasRoughnessMap) ? material.roughness : mr.g;
    }
    else {
        metallic = (!material.hasMetallicMap) ? material.metallic : texture(material.metallicMap, fsIn.TexCoord).r;
        roughness = (!material.hasRoughnessMap) ? material.roughness : texture(material.roughnessMap, fsIn.TexCoord).r;
    }
    metallic = material.metallicFactor * metallic;
    roughness = material.roughnessFactor * roughness;

    // Input lighting data
    vec3 N = getNormal();

    // Apply emissive component
    vec3 emissive = vec3(0.0);
    if (material.hasEmissiveMap) {
        emissive = material.emissiveFactor * texture(material.emissiveMap, fsIn.TexCoord).rgb;
    }

    // Apply ambient occlusion
    float ao = 1.0;
    if (material.hasAOMap) {
        ao = texture(material.aoMap, fsIn.TexCoord).r;
    }

    gAlbedo = albedo;
    gAlpha = alpha;
    gPBR = vec3(metallic, roughness, ao);
    gEmissive = vec4(emissive, material.IBL);
#ifdef VIEW_DEPENDENT_LIGHTING
    gNormal = vec3(N);
#else
    gNormal = normalize(fsIn.Normal);
#endif
    gPosition = fsIn.FragPosWorld;
    gLightPosition = fsIn.FragPosLightSpace;
    gIDs = uvec4(fsIn.DrawID, gl_PrimitiveID, 0, (alpha == 1.0) ? ALPHA_OPAQUE : ALPHA_BLEND);
    gIDs.z = floatBitsToUint((-fsIn.FragPosView.z - camera.near) / (camera.far - camera.near));
}
