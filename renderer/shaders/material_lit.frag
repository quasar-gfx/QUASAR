#include "constants.glsl"
#include "camera.glsl"
#include "pbr.glsl"
#include "depth_peeling.glsl"

layout(location = 0) out vec4 FragColor;
layout(location = 1) out float FragAlpha;
layout(location = 2) out vec3 FragNormal;
layout(location = 3) out uvec4 FragIDs;

in VertexData {
    flat uint DrawID;
    vec2 TexCoord;
    vec3 PositionView;
    vec3 PositionWorld;
    vec3 Color;
    vec3 Normal;
    vec3 Tangent;
    vec3 BiTangent;
    vec4 PositionLightSpace;
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

uniform AmbientLight ambientLight;
uniform DirectionalLight directionalLight;
layout(std140) uniform PointLightBlock {
    PointLight pointLights[MAX_POINT_LIGHTS];
    int numPointLights;
};

// Shadow maps
uniform sampler2D dirLightShadowMap; // 9
#ifndef GL_ES
uniform samplerCube pointLightShadowMaps[MAX_POINT_LIGHTS]; // 10+
#else
uniform samplerCube pointLightShadowMaps0; // 10
uniform samplerCube pointLightShadowMaps1; // 11
uniform samplerCube pointLightShadowMaps2; // 12
uniform samplerCube pointLightShadowMaps3; // 13
#endif

vec3 getNormal() {
	vec3 N = normalize(fsIn.Normal);
	vec3 T = normalize(fsIn.Tangent);
	vec3 B = normalize(fsIn.BiTangent);

    if (!material.hasNormalMap)
        return N;

    if (any(isnan(B))) {
        vec3 q1 = dFdx(fsIn.PositionWorld);
        vec3 q2 = dFdy(fsIn.PositionWorld);
        vec2 st1 = dFdx(fsIn.TexCoord);
        vec2 st2 = dFdy(fsIn.TexCoord);

        T = normalize(q1 * st2.t - q2 * st1.t);
        B = -normalize(cross(N, T));
    }

	mat3 TBN = mat3(T, B, N);
	vec3 tangentNormal = texture(material.normalMap, fsIn.TexCoord).xyz * 2.0 - 1.0;
	return normalize(TBN * tangentNormal);
}

// Depth peeling helpers moved to depth_peeling.glsl

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
    applyDepthPeeling(fsIn.PositionView);
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
    vec3 N = normalize(fsIn.Normal); // getNormal();
    vec3 V = normalize(camera.position - fsIn.PositionWorld);
    vec3 R = reflect(-V, N);

    // Calculate reflectance at normal incidence; if dia-electric (like plastic) use F0
    // Of 0.04 and if it's a metal, use the albedo baseColor as F0 (metallic workflow)
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);

    PBRInfo pbrInputs = PBRInfo(N, V, R, albedo, metallic, roughness, F0);

    // Apply reflectance equation for lights
    vec3 radianceOut = vec3(0.0);
    radianceOut += calcDirLight(directionalLight, pbrInputs, dirLightShadowMap, fsIn.PositionLightSpace, fsIn.Normal);
    for (int i = 0; i < numPointLights; i++) {
        PointLight light = pointLights[i];
#ifndef GL_ES
        radianceOut += calcPointLight(light, pointLightShadowMaps[light.shadowIndex], pbrInputs, fsIn.PositionWorld);
#else
             if (i == 0) radianceOut += calcPointLight(light, pointLightShadowMaps0, pbrInputs, fsIn.PositionWorld);
        else if (i == 1) radianceOut += calcPointLight(light, pointLightShadowMaps1, pbrInputs, fsIn.PositionWorld);
        else if (i == 2) radianceOut += calcPointLight(light, pointLightShadowMaps2, pbrInputs, fsIn.PositionWorld);
        else if (i == 3) radianceOut += calcPointLight(light, pointLightShadowMaps3, pbrInputs, fsIn.PositionWorld);
#endif
    }

    vec3 ambient = ambientLight.intensity * ambientLight.color * albedo;
    // Apply IBL
    ambient += material.IBL * calcIBLContribution(pbrInputs, material.irradianceMap, material.prefilterMap, material.brdfLUT);

    // Apply ambient occlusion
    if (material.hasAOMap) {
        float ao = texture(material.aoMap, fsIn.TexCoord).r;
        ambient *= ao;
    }
    radianceOut = radianceOut + ambient;

    // Apply emissive lighting
    if (material.hasEmissiveMap) {
        vec3 emissive = texture(material.emissiveMap, fsIn.TexCoord).rgb;
        radianceOut += material.emissiveFactor * emissive;
    }

    FragColor = vec4(radianceOut, alpha);
    FragAlpha = alpha;
    FragNormal = N;
    FragIDs = uvec4(fsIn.DrawID, gl_PrimitiveID, 0, (alpha == 1.0) ? ALPHA_OPAQUE : ALPHA_BLEND);
    FragIDs.z = floatBitsToUint((-fsIn.PositionView.z - camera.near) / (camera.far - camera.near));
}
