#include "constants.glsl"
#include "camera.glsl"
#include "pbr.glsl"

out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D gAlbedo; // 0
uniform sampler2D gPBR; // 1
uniform sampler2D gAlpha; // 2
uniform sampler2D gNormal; // 3
uniform sampler2D gPosition; // 4
uniform sampler2D gLightPosition; // 5

// Material
uniform struct Material {
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
uniform samplerCube pointLightShadowMaps[MAX_POINT_LIGHTS]; // 10+

void main() {
    vec4 albedo_er = texture(gAlbedo, TexCoords);
    vec4 mra_eg = texture(gPBR, TexCoords);
    vec2 alpha_eb = texture(gAlpha, TexCoords).rg;
    vec3 fragNormal = texture(gNormal, TexCoords).rgb;
    vec4 fragPosWorld_ibl = texture(gPosition, TexCoords);
    vec4 fragPosLightSpace = texture(gLightPosition, TexCoords);

    vec3 albedo = albedo_er.rgb;
    float alpha = alpha_eb.r;
    vec3 mra = mra_eg.rgb;
    float metallic = mra.r;
    float roughness = mra.g;
    float ao = mra.b;
    vec3 emissive = vec3(albedo_er.a, mra_eg.a, alpha_eb.g);

    float IBL = fragPosWorld_ibl.a;

    vec3 fragPosWorld = fragPosWorld_ibl.xyz;

    // Input lighting data
    vec3 N = fragNormal;
    vec3 V = normalize(camera.position - fragPosWorld);
    vec3 R = reflect(-V, N);

    // Calculate reflectance at normal incidence; if dia-electric (like plastic) use F0
    // Of 0.04 and if it's a metal, use the albedo baseColor as F0 (metallic workflow)
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);

    PBRInfo pbrInputs = PBRInfo(N, V, R, albedo, metallic, roughness, F0);

    // Apply reflectance equation for lights
    vec3 radianceOut = vec3(0.0);
    radianceOut += calcDirLight(directionalLight, pbrInputs, dirLightShadowMap, fragPosLightSpace, N);
    for (int i = 0; i < numPointLights; i++) {
        PointLight light = pointLights[i];
        radianceOut += calcPointLight(light, pointLightShadowMaps[light.shadowIndex], pbrInputs, fragPosWorld);
    }

    vec3 ambient = ambientLight.intensity * ambientLight.color * albedo;
    // Apply IBL
    ambient += IBL * calcIBLContribution(pbrInputs, material.irradianceMap, material.prefilterMap, material.brdfLUT);

    // Apply emissive component
    radianceOut += emissive;

    // Apply ambient occlusion
    ambient *= ao;

    radianceOut = radianceOut + ambient;

    FragColor = vec4(radianceOut, 1.0); // set alpha to 1.0 for now
}
