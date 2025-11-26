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

vec2 poissonDisk[16] = vec2[](
    vec2(-0.94201624, -0.39906216), vec2( 0.94558609, -0.76890725),
    vec2(-0.09418410, -0.92938870), vec2( 0.34495938,  0.29387760),
    vec2(-0.91588581,  0.45771432), vec2(-0.81544232, -0.87912464),
    vec2(-0.38277543,  0.27676845), vec2( 0.97484398,  0.75648379),
    vec2( 0.44323325, -0.97511554), vec2( 0.53742981, -0.47373420),
    vec2(-0.26496911, -0.41893023), vec2( 0.79197514,  0.19090188),
    vec2(-0.24188840,  0.99706507), vec2(-0.81409955,  0.91437590),
    vec2( 0.19984126,  0.78641367), vec2( 0.14383161, -0.14100790)
);

// Based on http://byteblacksmith.com/improvements-to-the-canonical-one-liner-glsl-rand-for-opengl-es-2-0/
float random(vec2 co) {
    float a  = 12.9898;
    float b  = 78.233;
    float c  = 43758.5453;
    float dt = dot(co.xy ,vec2(a,b));
    float sn = mod(dt,3.14);
    return fract(sin(sn) * c);
}

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
float calcDirLightShadow(DirectionalLight light, sampler2D dirLightShadowMap, vec4 fragPositionLightSpace, vec3 fragNormal) {
    float shadowFactor = 0.0;
    int samples = 16;

    vec3 projCoords = fragPositionLightSpace.xyz / fragPositionLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;
    if (projCoords.z > 1.0 || projCoords.x > 1.0 || projCoords.x < 0.0 || projCoords.y > 1.0 || projCoords.y < 0.0) {
        return 1.0; // outside the shadow map, assume lit
    }

    float cosTheta = clamp(dot(normalize(fragNormal), normalize(light.direction)), 0.0, 1.0);
    float bias = max(0.005 * (1.0 - cosTheta), 0.001);

    vec2 texelSize = 1.0 / vec2(textureSize(dirLightShadowMap, 0));

    // Rotate the kernel
    float noise = random(gl_FragCoord.xy);
    float angle = noise * (2.0 * PI);
    float s = sin(angle);
    float c = cos(angle);
    mat2 rotation = mat2(c, -s, s, c);

    float diskRadius = 2.0;
    for(int i = 0; i < samples; i++) {
        // Rotate the Poisson offset
        vec2 offset = (rotation * poissonDisk[i]) * diskRadius * texelSize;

        // Sample the shadow map (closest depth)
        float closestDepth = texture(dirLightShadowMap, projCoords.xy + offset).r;
        // Check if current fragment is behind the closest depth
        float currentDepth = projCoords.z;
        if (currentDepth - bias > closestDepth) {
            shadowFactor += 1.0; // blocked
        }
    }

    shadowFactor /= float(samples);
    return 1.0 - shadowFactor; // 0.0 is blocked, 1.0 is lit
}

// Directional light with Filament BRDF
vec3 calcDirLight(DirectionalLight light, PBRInfo pbrInputs, sampler2D dirLightShadowMap, vec4 fragPositionLightSpace, vec3 fragNormal) {
    if (light.intensity == 0.0) return vec3(0.0);

    vec3 L = normalize(-light.direction);
    vec3 radianceIn = light.color * light.intensity;

    // Returns 1.0 if lit, 0.0 if dark
    float visibility = calcDirLightShadow(light, dirLightShadowMap, fragPositionLightSpace, fragNormal);
    vec3 brdf = computeBRDF(pbrInputs, L, radianceIn);
    return brdf * visibility;
}

// Point light shadow using cube map
float calcPointLightShadows(PointLight light, samplerCube pointLightShadowMap, vec3 fragToLight, vec3 fragPositionWorld) {
    float shadowFactor = 0.0;
    int samples = 16;
    float bias = 0.05;

    vec3 lightDir = normalize(fragToLight);
    vec3 right = cross(lightDir, vec3(0.0, 1.0, 0.0));
    if (length(right) < 0.001) {
        right = cross(lightDir, vec3(1.0, 0.0, 0.0));
    }
    right = normalize(right);
    vec3 up = cross(right, lightDir);

    float noise = random(gl_FragCoord.xy);
    float angle = noise * (2.0 * PI);
    float s = sin(angle);
    float c = cos(angle);

    float diskRadius = 0.05;
    float currentDepth = length(fragToLight);
    for (int i = 0; i < samples; i++) {
        // Rotate the Poisson offset
        vec2 offset;
        offset.x = (c * poissonDisk[i].x - s * poissonDisk[i].y);
        offset.y = (s * poissonDisk[i].x + c * poissonDisk[i].y);

        vec3 sampleDir = lightDir + (right * offset.x * diskRadius) + (up * offset.y * diskRadius);

        float closestDepth = texture(pointLightShadowMap, sampleDir).r;
        closestDepth *= light.shadowFar;
        if (currentDepth - bias < closestDepth) {
            shadowFactor += 1.0; // blocked
        }
    }

    shadowFactor /= float(samples);
    return shadowFactor; // 0.0 is blocked, 1.0 is lit
}

// Point light with Filament BRDF
vec3 calcPointLight(PointLight light, samplerCube pointLightShadowMap, PBRInfo pbrInputs, vec3 fragPositionWorld) {
    if (light.intensity == 0.0) return vec3(0.0);

    vec3 L = normalize(light.position - fragPositionWorld);
    float distance = length(light.position - fragPositionWorld);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * distance * distance);
    vec3 radianceIn = light.color * light.intensity * attenuation;

    vec3 fragToLight = fragPositionWorld - light.position;

    // Returns 1.0 if lit, 0.0 if dark
    float visibility = calcPointLightShadows(light, pointLightShadowMap, fragToLight, fragPositionWorld);
    vec3 brdf = computeBRDF(pbrInputs, L, radianceIn);
    return brdf * visibility;
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
