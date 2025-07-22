#include "constants.glsl"
#include "lights.glsl"

struct PBRInfo {
    vec3 N;
    vec3 V;
    vec3 R;
    vec3 albedo;
    float metallic;
    float roughness;
    vec3 F0;
};

vec3 gridSamplingDisk[20] = vec3[]
(
   vec3( 1,  1,  1), vec3( 1, -1,  1), vec3(-1, -1,  1), vec3(-1,  1,  1),
   vec3( 1,  1, -1), vec3( 1, -1, -1), vec3(-1, -1, -1), vec3(-1,  1, -1),
   vec3( 1,  1,  0), vec3( 1, -1,  0), vec3(-1, -1,  0), vec3(-1,  1,  0),
   vec3( 1,  0,  1), vec3(-1,  0,  1), vec3( 1,  0, -1), vec3(-1,  0, -1),
   vec3( 0,  1,  1), vec3( 0, -1,  1), vec3( 0, -1, -1), vec3( 0,  1, -1)
);

// GGX Normal Distribution
float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    return a2 / (PI * denom * denom);
}

// Schlick's approximation for geometry
float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

// Smith's method for combined geometry
float GeometrySmith(float NdotV, float NdotL, float roughness) {
    return GeometrySchlickGGX(NdotV, roughness) * GeometrySchlickGGX(NdotL, roughness);
}

// Filament's Disney Diffuse
float DiffuseBurley(float NdotV, float NdotL, float LdotH, float roughness) {
    float fd90 = 0.5 + 2.0 * LdotH * LdotH * roughness;
    float lightScatter = 1.0 + (fd90 - 1.0) * pow(1.0 - NdotL, 5.0);
    float viewScatter = 1.0 + (fd90 - 1.0) * pow(1.0 - NdotV, 5.0);
    return lightScatter * viewScatter;
}

// Fresnel with energy compensation
vec3 FresnelSchlick(float cosTheta, vec3 F0, float roughness) {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0);
}

// Compute the BRDF with optional view-dependent lighting
vec3 computeBRDF(PBRInfo pbrInputs, vec3 L, vec3 radianceIn) {
    vec3 N = pbrInputs.N;
    vec3 V = pbrInputs.V;
    vec3 H = normalize(V + L);
    vec3 albedo = pbrInputs.albedo;
    float metallic = pbrInputs.metallic;
    float roughness = pbrInputs.roughness;
    vec3 F0 = pbrInputs.F0;

    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.0);
    float NdotH = max(dot(N, H), 0.0);
    float LdotH = max(dot(L, H), 0.0);

    float D = DistributionGGX(N, H, roughness);
    float G = GeometrySmith(NdotV, NdotL, roughness);

    vec3 F = vec3(0.0);
#ifdef VIEW_DEPENDENT_LIGHTING
    F = FresnelSchlick(LdotH, F0, roughness);
    vec3 specular = (D * G * F) / max(4.0 * NdotL * NdotV, 0.001);
#else
    vec3 specular = vec3(0.0);
#endif

    float energyComp = 1.0 + roughness;
    vec3 kS = F * energyComp;
    vec3 kD = (1.0 - kS) * (1.0 - metallic);

    float disneyDiffuse = DiffuseBurley(NdotV, NdotL, LdotH, roughness);
    vec3 diffuse = (albedo / PI) * disneyDiffuse;

    return (kD * diffuse + specular) * radianceIn * NdotL;
}

// Shadow calculation for directional light
float calcDirLightShadow(DirectionalLight light, sampler2D dirLightShadowMap, vec4 fragPosLightSpace, vec3 fragNormal) {
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;

    int samples = 9;
    float shadow = 0.0;
    vec3 normal = normalize(fragNormal);
    vec3 lightDir = normalize(-light.direction);
    float bias = max(0.05 * (1.0 - max(dot(normal, lightDir), 0.0)), 0.005);
    vec2 texelSize = 1.0 / vec2(textureSize(dirLightShadowMap, 0));

    for (int i = 0; i < samples; i++) {
        float pcfDepth = texture(dirLightShadowMap, projCoords.xy + gridSamplingDisk[i].xy * texelSize).r;
        shadow += projCoords.z - bias > pcfDepth ? 0.111111 : 0.0;
    }
    return shadow;
}

// Point light shadow using cube map
float calcPointLightShadows(PointLight light, samplerCube pointLightShadowMap, vec3 fragToLight, vec3 fragPosWorld) {
    float currentDepth = length(fragToLight);
    int samples = 20;
    float shadow = 0.0;
    float bias = 0.15;
    float viewDistance = length(camera.position - fragPosWorld);
    float diskRadius = (1.0 + (viewDistance / light.farPlane)) / 25.0;

    for (int i = 0; i < samples; i++) {
        float closestDepth = texture(pointLightShadowMap, fragToLight + gridSamplingDisk[i] * diskRadius).r;
        closestDepth *= light.farPlane;
        if (currentDepth - bias > closestDepth)
            shadow += 1.0;
    }
    shadow /= float(samples);
    return shadow;
}

// Directional light with Filament BRDF
vec3 calcDirLight(DirectionalLight light, PBRInfo pbrInputs, sampler2D dirLightShadowMap, vec4 fragPosLightSpace, vec3 fragNormal) {
    if (light.intensity == 0.0) return vec3(0.0);

    vec3 L = normalize(-light.direction);
    vec3 radianceIn = light.color * light.intensity;

    float shadow = calcDirLightShadow(light, dirLightShadowMap, fragPosLightSpace, fragNormal);
    vec3 brdf = computeBRDF(pbrInputs, L, radianceIn);
    return brdf * (1.0 - shadow);
}

// Point light with Filament BRDF
vec3 calcPointLight(PointLight light, samplerCube pointLightShadowMap, PBRInfo pbrInputs, vec3 fragPosWorld) {
    if (light.intensity == 0.0) return vec3(0.0);

    vec3 L = normalize(light.position - fragPosWorld);
    float distance = length(light.position - fragPosWorld);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * distance * distance);
    vec3 radianceIn = light.color * light.intensity * attenuation;

    vec3 fragToLight = fragPosWorld - light.position;
    float shadow = calcPointLightShadows(light, pointLightShadowMap, fragToLight, fragPosWorld);
    vec3 brdf = computeBRDF(pbrInputs, L, radianceIn);
    return brdf * (1.0 - shadow);
}

vec3 calcIBLContribution(PBRInfo pbrInputs, samplerCube irradianceMap, samplerCube prefilterMap, sampler2D brdfLUT) {
#ifdef VIEW_DEPENDENT_LIGHTING
    vec3 N = pbrInputs.N;
    vec3 V = pbrInputs.V;
    vec3 R = pbrInputs.R;
    vec3 albedo = pbrInputs.albedo;
    float metallic = pbrInputs.metallic;
    float roughness = pbrInputs.roughness;
    vec3 F0 = pbrInputs.F0;

    vec3 kS = FresnelSchlick(max(dot(N, V), 0.0), F0, roughness);
    vec3 kD = (1.0 - kS) * (1.0 - metallic);

    vec3 irradiance = texture(irradianceMap, N).rgb;
    vec3 diffuse = irradiance * albedo;

    const float MAX_REFLECTION_LOD = 4.0;
    vec3 prefilteredColor = textureLod(prefilterMap, R, roughness * MAX_REFLECTION_LOD).rgb;
    vec2 brdf = texture(brdfLUT, vec2(max(dot(N, V), 0.0), roughness)).rg;
    vec3 specular = prefilteredColor * (kS * brdf.x + brdf.y);

    return kD * diffuse + specular;
#else
    vec3 irradiance = texture(irradianceMap, pbrInputs.N).rgb;
    vec3 diffuse = irradiance * pbrInputs.albedo * (1.0 - pbrInputs.metallic);
    return diffuse;
#endif
}
