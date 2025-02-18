#include "constants.glsl"
#include "camera.glsl"
#include "pbr.glsl"

out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D gAlbedo; // 0
uniform sampler2D gPBR; // 1
uniform sampler2D gEmissive; // 2
uniform sampler2D gLightPositionXYZ; // 3
uniform sampler2D gLightPositionWIBLAlpha; // 4
uniform sampler2D gPosition; // 5
uniform sampler2D gNormal; // 6

// material
uniform struct Material {
#ifdef PLATFORM_CORE
    samplerCube irradianceMap; // 7
    samplerCube prefilterMap; // 8
    sampler2D brdfLUT; // 9
#endif
} material;

#define MAX_POINT_LIGHTS 4

uniform AmbientLight ambientLight;
uniform DirectionalLight directionalLight;
uniform PointLight pointLights[MAX_POINT_LIGHTS];
uniform int numPointLights;

// shadow maps
uniform sampler2D dirLightShadowMap; // 9
#ifdef PLATFORM_CORE
uniform samplerCube pointLightShadowMaps[MAX_POINT_LIGHTS]; // 10+
#endif

void main() {
    vec3 albedo = texture(gAlbedo, TexCoords).rgb;

    vec3 mra = texture(gPBR, TexCoords).rgb;
    float metallic = mra.r;
    float roughness = mra.g;
    float ao = mra.b;
    vec3 emissive = texture(gEmissive, TexCoords).rgb;

    vec4 fragPosLightSpace;
    fragPosLightSpace.xyz = texture(gLightPositionXYZ, TexCoords).rgb;

    vec3 wia = texture(gLightPositionWIBLAlpha, TexCoords).rgb;
    fragPosLightSpace.w = wia.r;
    float IBL = wia.g;
    float alpha = wia.b;

    vec3 fragPosWorld = texture(gPosition, TexCoords).rgb;
    vec3 fragNormal = texture(gNormal, TexCoords).rgb;

    // input lighting data
    vec3 N = fragNormal;
    vec3 V = normalize(camera.position - fragPosWorld);
    vec3 R = reflect(-V, N);

    // calculate reflectance at normal incidence; if dia-electric (like plastic) use F0
    // of 0.04 and if it's a metal, use the albedo baseColor as F0 (metallic workflow)
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);

    PBRInfo pbrInputs = PBRInfo(N, V, R, albedo, metallic, roughness, F0);

    // apply reflectance equation for lights
    vec3 radianceOut = vec3(0.0);
    radianceOut += calcDirLight(directionalLight, pbrInputs, dirLightShadowMap, fragPosLightSpace, N);
    for (int i = 0; i < numPointLights; i++) {
#ifdef PLATFORM_CORE
        radianceOut += calcPointLight(pointLights[i], pointLightShadowMaps[i], pbrInputs, fragPosWorld);
#else
        radianceOut += calcPointLight(pointLights[i], pbrInputs, fragPosWorld);
#endif
    }

    vec3 ambient = ambientLight.intensity * ambientLight.color * albedo;
#ifdef PLATFORM_CORE
    // apply IBL
    ambient += IBL * getIBLContribution(pbrInputs, material.irradianceMap, material.prefilterMap, material.brdfLUT);
#endif

    // apply emissive component
    radianceOut += emissive;

    // apply ambient occlusion
    ambient *= ao;

    radianceOut = radianceOut + ambient;

    FragColor = vec4(radianceOut, 1.0);
}
