#include "lights.glsl"

vec3 gridSamplingDisk[20] = vec3[]
(
   vec3( 1,  1,  1), vec3( 1, -1,  1), vec3(-1, -1,  1), vec3(-1,  1,  1),
   vec3( 1,  1, -1), vec3( 1, -1, -1), vec3(-1, -1, -1), vec3(-1,  1, -1),
   vec3( 1,  1,  0), vec3( 1, -1,  0), vec3(-1, -1,  0), vec3(-1,  1,  0),
   vec3( 1,  0,  1), vec3(-1,  0,  1), vec3( 1,  0, -1), vec3(-1,  0, -1),
   vec3( 0,  1,  1), vec3( 0, -1,  1), vec3( 0, -1, -1), vec3( 0,  1, -1)
);

float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness*roughness;
    float a2 = a*a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float num   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    return num / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float num   = NdotV;
    float denom = NdotV * (1.0 - k) + k;
    return num / denom;
}

float GeometrySmith(float NdotV, float NdotL, float roughness) {
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    float val = 1.0 - cosTheta;
    return F0 + (1.0 - F0) * (val*val*val*val*val); // val^5
}

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    float val = 1.0 - cosTheta;
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * (val*val*val*val*val); // val^5
}

#ifdef PLATFORM_CORE
vec3 getIBLContribution(PBRInfo pbrInputs, samplerCube irradianceMap, samplerCube prefilterMap, sampler2D brdfLUT) {
    vec3 N = pbrInputs.N;
    vec3 V = pbrInputs.V;
    vec3 R = pbrInputs.R;
    vec3 albedo = pbrInputs.albedo;
    float metallic = pbrInputs.metallic;
    float roughness = pbrInputs.roughness;
    vec3 F0 = pbrInputs.F0;

    vec3 kS = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - metallic;

    vec3 irradiance = texture(irradianceMap, N).rgb;
    vec3 diffuse = irradiance * albedo;

    // sample both the pre-filter map and the BRDF lut and combine them together as per the Split-Sum approximation
    // to get the IBL specular part
    const float MAX_REFLECTION_LOD = 4.0;
    vec3 prefilteredColor = textureLod(prefilterMap, R, roughness * MAX_REFLECTION_LOD).rgb;
    vec2 brdf = texture(brdfLUT, vec2(max(dot(N, V), 0.0), roughness)).rg;
#ifdef VIEW_DEPENDENT_LIGHTING
    vec3 specular = prefilteredColor * (kS * brdf.x + brdf.y);
#else
    vec3 specular = vec3(0.0);
#endif
    return kD * diffuse + specular;
}
#else
vec3 getIBLContribution(PBRInfo pbrInputs) {
    return vec3(0.0);
}
#endif

float calcDirLightShadow(DirectionalLight light, sampler2D dirLightShadowMap, vec4 fragPosLightSpace, vec3 fragNormal) {
    // perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;

    // PCF
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

float calcPointLightShadows(PointLight light, samplerCube pointLightShadowMap, vec3 fragToLight, vec3 fragPosWorld) {
    float currentDepth = length(fragToLight);

    // PCF
    int samples = 20;
    float shadow = 0.0;
    float bias = 0.15;
    float viewDistance = length(camera.position - fragPosWorld);
    float diskRadius = (1.0 + (viewDistance / light.farPlane)) / 25.0;
    for (int i = 0; i < samples; i++) {
        float closestDepth = texture(pointLightShadowMap, fragToLight + gridSamplingDisk[i] * diskRadius).r;
        closestDepth *= light.farPlane;   // undo mapping [0;1]
        if (currentDepth - bias > closestDepth)
            shadow += 1.0;
    }
    shadow /= float(samples);

    return shadow;
}

vec3 calcDirLight(DirectionalLight light, PBRInfo pbrInputs, sampler2D dirLightShadowMap, vec4 fragPosLightSpace, vec3 fragNormal) {
    if (light.intensity == 0.0)
        return vec3(0.0);

    vec3 N = pbrInputs.N;
    vec3 V = pbrInputs.V;
    vec3 R = pbrInputs.R;
    vec3 albedo = pbrInputs.albedo;
    float metallic = pbrInputs.metallic;
    float roughness = pbrInputs.roughness;
    vec3 F0 = pbrInputs.F0;

    // calculate per-light radiance
    vec3 L = normalize(-light.direction);
    vec3 H = normalize(V + L);
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    vec3 radianceIn = light.intensity * light.color;

    // Cook-Torrance BRDF
    float NDF = DistributionGGX(N, H, roughness);
    float G   = GeometrySmith(NdotV, NdotL, roughness);
    vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);

#ifdef VIEW_DEPENDENT_LIGHTING
    vec3 numerator    = NDF * G * F;
    float denominator = 4.0 * NdotV * NdotL + 0.0001; // + 0.0001 to prevent divide by zero
    vec3 specular = numerator / denominator;
#else
    vec3 specular = vec3(0.0);
#endif

    // kS is equal to Fresnel
    vec3 kS = F;
    // for energy conservation, the diffuse and specular light can't
    // be above 1.0 (unless the surface emits light); to preserve this
    // relationship the diffuse component (kD) should equal 1.0 - kS.
    vec3 kD = vec3(1.0) - kS;
    // multiply kD by the inverse metalness such that only non-metals
    // have diffuse lighting, or a linear blend if partly metal (pure metals
    // have no diffuse light).
    kD *= 1.0 - metallic;

    // add to outgoing radiance Lo
    vec3 radianceOut = (kD * (albedo / PI) + specular) * radianceIn * NdotL; // note that we already multiplied the BRDF by the Fresnel (kS) so we won't multiply by kS again

    // shadow calcs
    float shadow = calcDirLightShadow(light, dirLightShadowMap, fragPosLightSpace, fragNormal);
    radianceOut *= (1.0 - shadow);

    return radianceOut;
}

#ifdef PLATFORM_CORE
vec3 calcPointLight(PointLight light, samplerCube pointLightShadowMap, PBRInfo pbrInputs, vec3 fragPosWorld) {
#else
vec3 calcPointLight(PointLight light, PBRInfo pbrInputs, vec3 fragPosWorld) {
#endif
    if (light.intensity == 0.0)
        return vec3(0.0);

    vec3 N = pbrInputs.N;
    vec3 V = pbrInputs.V;
    vec3 R = pbrInputs.R;
    vec3 albedo = pbrInputs.albedo;
    float metallic = pbrInputs.metallic;
    float roughness = pbrInputs.roughness;
    vec3 F0 = pbrInputs.F0;

    // calculate per-light radiance
    vec3 L = normalize(light.position - fragPosWorld);
    vec3 H = normalize(V + L);
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float distance = length(light.position - fragPosWorld);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
    vec3 radianceIn = light.intensity * light.color * attenuation;

    // Cook-Torrance BRDF
    float NDF = DistributionGGX(N, H, roughness);
    float G   = GeometrySmith(NdotV, NdotL, roughness);
    vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);

#ifdef VIEW_DEPENDENT_LIGHTING
    vec3 numerator    = NDF * G * F;
    float denominator = 4.0 * NdotV * NdotL + 0.0001; // + 0.0001 to prevent divide by zero
    vec3 specular = numerator / denominator;
#else
    vec3 specular = vec3(0.0);
#endif

    // kS is equal to Fresnel
    vec3 kS = F;
    // for energy conservation, the diffuse and specular light can't
    // be above 1.0 (unless the surface emits light); to preserve this
    // relationship the diffuse component (kD) should equal 1.0 - kS.
    vec3 kD = vec3(1.0) - kS;
    // multiply kD by the inverse metalness such that only non-metals
    // have diffuse lighting, or a linear blend if partly metal (pure metals
    // have no diffuse light).
    kD *= 1.0 - metallic;

    // add to outgoing radiance Lo
    vec3 radianceOut = (kD * (albedo / PI) + specular) * radianceIn * NdotL; // note that we already multiplied the BRDF by the Fresnel (kS) so we won't multiply by kS again

    // shadow calcs
    vec3 fragToLight = fragPosWorld - light.position;
#ifdef PLATFORM_CORE
    float shadow = calcPointLightShadows(light, pointLightShadowMap, fragToLight, fragPosWorld);
    radianceOut *= (1.0 - shadow);
#endif

    return radianceOut;
}
