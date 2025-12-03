#include "constants.glsl"
#include "camera.glsl"
#include "pbr.glsl"

layout(location = 0) out vec4 FragColor;
layout(location = 1) out float FragAlpha;
layout(location = 2) out vec3 FragNormal;
layout(location = 3) out uvec4 FragIDs;

in vec2 TexCoord;

uniform sampler2D gAlbedo; // 0
uniform sampler2D gAlpha; // 1
uniform sampler2D gPBR; // 2
uniform sampler2D gEmissive; // 3
uniform sampler2D gNormal; // 4
uniform sampler2D gPosition; // 5
uniform sampler2D gLightPosition; // 6

// Material
uniform struct Material {
    samplerCube irradianceMap; // 7
    samplerCube prefilterMap; // 8
    sampler2D brdfLUT; // 9
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

void main() {
    vec3 albedo = texture(gAlbedo, TexCoord).rgb;
    float alpha = texture(gAlpha, TexCoord).r;
    vec3 mra = texture(gPBR, TexCoord).rgb;
    vec4 emissive_IBL = texture(gEmissive, TexCoord);
    vec3 fragNormal = texture(gNormal, TexCoord).rgb;
    vec3 PositionWorld = texture(gPosition, TexCoord).xyz;
    vec4 PositionLightSpace = texture(gLightPosition, TexCoord);

    float metallic = mra.r;
    float roughness = mra.g;
    float ao = mra.b;

    vec3 emissive = emissive_IBL.rgb;
    float IBL = emissive_IBL.a;

    // Input lighting data
    vec3 N = fragNormal;
    vec3 V = normalize(camera.position - PositionWorld);
    vec3 R = reflect(-V, N);

    // Calculate reflectance at normal incidence; if dia-electric (like plastic) use F0
    // Of 0.04 and if it's a metal, use the albedo baseColor as F0 (metallic workflow)
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);

    PBRInfo pbrInputs = PBRInfo(N, V, R, albedo, metallic, roughness, F0);

    // Direct lighting
    vec3 radianceOut = vec3(0.0);
    radianceOut += calcDirLight(directionalLight, pbrInputs, dirLightShadowMap, PositionLightSpace, N);
    for (int i = 0; i < numPointLights; i++) {
        PointLight light = pointLights[i];
#ifndef GL_ES
        radianceOut += calcPointLight(light, pointLightShadowMaps[light.shadowIndex], pbrInputs, PositionWorld);
#else
             if (i == 0) radianceOut += calcPointLight(light, pointLightShadowMaps0, pbrInputs, PositionWorld);
        else if (i == 1) radianceOut += calcPointLight(light, pointLightShadowMaps1, pbrInputs, PositionWorld);
        else if (i == 2) radianceOut += calcPointLight(light, pointLightShadowMaps2, pbrInputs, PositionWorld);
        else if (i == 3) radianceOut += calcPointLight(light, pointLightShadowMaps3, pbrInputs, PositionWorld);
#endif
    }

    vec3 ambient = ambientLight.intensity * ambientLight.color * albedo;
    // Apply IBL
    ambient += IBL * calcIBLContribution(pbrInputs, material.irradianceMap, material.prefilterMap, material.brdfLUT);

    // Apply ambient occlusion
    ambient *= ao;

    // Apply emissive lighting
    radianceOut += emissive;

    radianceOut = radianceOut + ambient;

    FragColor = vec4(radianceOut, alpha);
    FragAlpha = alpha;
    FragNormal = N;
}
