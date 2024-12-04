layout(location = 0) out vec4 FragColor;
layout(location = 1) out vec4 FragPosition;
layout(location = 2) out vec4 FragNormal;
layout(location = 3) out uvec4 FragIDs;

in VertexData {
    flat uint drawID;
    vec2 TexCoords;
    vec3 FragPos;
    vec3 FragPosView;
    vec3 Color;
    vec3 Normal;
    vec3 Tangent;
    vec3 BiTangent;
    vec4 FragPosLightSpace;
} fsIn;

// material
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

    // material textures
    sampler2D baseColorMap; // 0
    sampler2D normalMap; // 1
    sampler2D metallicMap; // 2
    sampler2D roughnessMap; // 3
    sampler2D aoMap; // 4
    sampler2D emissiveMap; // 5

    // IBL
    float IBL; // IBL contribution
#ifdef PLATFORM_CORE
    samplerCube irradianceMap; // 6
    samplerCube prefilterMap; // 7
    sampler2D brdfLUT; // 8
#endif
} material;

// lights
struct AmbientLight {
    vec3 color;
    float intensity;
};

struct DirectionalLight {
    vec3 color;
    vec3 direction;
    float intensity;
};

struct PointLight {
    vec3 color;
    vec3 position;
    float intensity;
    float constant;
    float linear;
    float quadratic;
    float farPlane;
};

struct PBRInfo {
    vec3 N;
    vec3 V;
    vec3 R;
    vec3 albedo;
    float metallic;
    float roughness;
    vec3 F0;
};

uniform int numPointLights;
uniform AmbientLight ambientLight;
uniform DirectionalLight directionalLight;
uniform PointLight pointLights[MAX_POINT_LIGHTS];

// shadow maps
uniform sampler2D dirLightShadowMap; // 9
#ifdef PLATFORM_CORE
uniform samplerCube pointLightShadowMaps[MAX_POINT_LIGHTS]; // 10+
#endif

uniform struct Camera {
#ifndef ANDROID
    mat4 projection;
    mat4 view;
#else
    mat4 projection[2];
    mat4 view[2];
#endif
    vec3 position;
    float fovy;
    float near;
    float far;
} camera;

#ifdef DO_DEPTH_PEELING
uniform bool peelDepth;
uniform usampler2D prevDepthMap;

uniform int height;
uniform float E;
uniform float edpDelta;
#endif

#define MAX_DEPTH 0.9999
const float PI = 3.1415926535897932384626433832795;

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

vec3 getNormal() {
#ifdef VIEW_DEPENDENT_LIGHTING
	vec3 N = normalize(fsIn.Normal);
	vec3 T = normalize(fsIn.Tangent);
	vec3 B = normalize(fsIn.BiTangent);

    if (!material.hasNormalMap)
        return N;

    // HACK: sometimes bitangent is nan, so recompute it
    if (any(isnan(B))) {
        B = normalize(cross(T, N));
        T = normalize(cross(N, B));
    }

	mat3 TBN = mat3(T, B, N);
	vec3 tangentNormal = normalize(texture(material.normalMap, fsIn.TexCoords).xyz * 2.0 - 1.0);
	return normalize(TBN * tangentNormal);
#else
    return normalize(fsIn.Normal);
#endif
}

vec3 getIBLContribution(PBRInfo pbrInputs) {
#if defined(PLATFORM_CORE) && defined(VIEW_DEPENDENT_LIGHTING)
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

    vec3 irradiance = texture(material.irradianceMap, N).rgb;
    vec3 diffuse = irradiance * albedo;

    // sample both the pre-filter map and the BRDF lut and combine them together as per the Split-Sum approximation
    // to get the IBL specular part
    const float MAX_REFLECTION_LOD = 4.0;
    vec3 prefilteredColor = textureLod(material.prefilterMap, R, roughness * MAX_REFLECTION_LOD).rgb;
    vec2 brdf = texture(material.brdfLUT, vec2(max(dot(N, V), 0.0), roughness)).rg;
    vec3 specular = prefilteredColor * (kS * brdf.x + brdf.y);
    return kD * diffuse + specular;
#else
    return vec3(0.0);
#endif
}

float calcDirLightShadow(DirectionalLight light, vec4 fragPosLightSpace) {
    // perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;

    // PCF
    int samples = 9;
    float shadow = 0.0;
    vec3 normal = normalize(fsIn.Normal);
    vec3 lightDir = normalize(-light.direction);
    float bias = max(0.05 * (1.0 - max(dot(normal, lightDir), 0.0)), 0.005);
    vec2 texelSize = 1.0 / vec2(textureSize(dirLightShadowMap, 0));
    for (int i = 0; i < samples; i++) {
        float pcfDepth = texture(dirLightShadowMap, projCoords.xy + gridSamplingDisk[i].xy * texelSize).r;
        shadow += projCoords.z - bias > pcfDepth ? 0.111111 : 0.0;
    }

    return shadow;
}

float calcPointLightShadows(PointLight light, samplerCube pointLightShadowMap, vec3 fragToLight) {
    float currentDepth = length(fragToLight);

    // PCF
    int samples = 20;
    float shadow = 0.0;
    float bias = 0.15;
    float viewDistance = length(camera.position - fsIn.FragPos);
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

vec3 calcDirLight(DirectionalLight light, PBRInfo pbrInputs) {
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

    vec3 numerator    = NDF * G * F;
    float denominator = 4.0 * NdotV * NdotL;
    vec3 specular = numerator / max(denominator, 0.0001);

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
    float shadow = calcDirLightShadow(light, fsIn.FragPosLightSpace);
    radianceOut *= (1.0 - shadow);

    return radianceOut;
}

#ifdef PLATFORM_CORE
vec3 calcPointLight(PointLight light, samplerCube pointLightShadowMap, PBRInfo pbrInputs) {
#else
vec3 calcPointLight(PointLight light, PBRInfo pbrInputs) {
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
    vec3 L = normalize(light.position - fsIn.FragPos);
    vec3 H = normalize(V + L);
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float distance = length(light.position - fsIn.FragPos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
    vec3 radianceIn = light.intensity * light.color * attenuation;

    // Cook-Torrance BRDF
    float NDF = DistributionGGX(N, H, roughness);
    float G   = GeometrySmith(NdotV, NdotL, roughness);
    vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 numerator    = NDF * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001; // + 0.0001 to prevent divide by zero
    vec3 specular = numerator / denominator;

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
    vec3 fragToLight = fsIn.FragPos - light.position;
#ifdef PLATFORM_CORE
    float shadow = calcPointLightShadows(light, pointLightShadowMap, fragToLight);
    radianceOut *= (1.0 - shadow);
#endif

    return radianceOut;
}

#ifdef DO_DEPTH_PEELING

#define EDP_SAMPLES 16

float LCOC(float d, float df) {
	float K = float(height)*0.5 / df / tan(camera.fovy*0.5); // screen-space LCOC scale
	return K * E * abs(df-d) / d; // relative radius of COC against df (blocker depth)
}

bool inPVHV(ivec2 pixelCoords, vec3 fragViewPos, uvec4 q) {
    float fragmentDepth = -fragViewPos.z;

    uint q_item = q.r;
    if (q_item < 0) return false;

    float blockerDepthNormalized = uintBitsToFloat(q.z);
	float df = mix(camera.near, camera.far, blockerDepthNormalized);
    float R = LCOC(fragmentDepth, df);
    for (int i = 0; i < EDP_SAMPLES; i++) {
        // sample around a circle with radius R
        float x = R * cos(float(i) * 2*PI / EDP_SAMPLES);
        float y = R * sin(float(i) * 2*PI / EDP_SAMPLES);
        vec2 offset = vec2(x, y);

        uvec4 w = texelFetch(prevDepthMap, ivec2(round(vec2(pixelCoords) + offset)), 0);
        uint w_item = w.r;
        if (w_item < 0) return true;

        float sampleDepthNormalized = uintBitsToFloat(w.z);
        if (sampleDepthNormalized == 0) return true;
        if (sampleDepthNormalized >= MAX_DEPTH) continue;

        if (q_item != w_item) return true;
        else if (sampleDepthNormalized >= blockerDepthNormalized + edpDelta) return true;
        else if (sampleDepthNormalized <= blockerDepthNormalized - edpDelta) return true;
    }

    return false;
}
#endif

void main() {
#ifdef DO_DEPTH_PEELING
    if (peelDepth) {
        ivec2 pixelCoords = ivec2(gl_FragCoord.xy);
        uvec4 q = texelFetch(prevDepthMap, pixelCoords, 0);

        float currDepth = (-fsIn.FragPosView.z - camera.near) / (camera.far - camera.near);
        float prevDepth = uintBitsToFloat(q.z);
        if (prevDepth == 0 || prevDepth >= MAX_DEPTH)
            discard;
        if (currDepth <= prevDepth)
            discard;
#ifdef EDP
        vec3 fragViewPos = fsIn.FragPosView;
        if (!inPVHV(pixelCoords, fragViewPos, q))
            discard;
#endif
    }
#endif

    vec4 baseColor;
    if (material.hasBaseColorMap) {
        baseColor = texture(material.baseColorMap, fsIn.TexCoords) * material.baseColorFactor;
    }
    else {
        baseColor = material.baseColorFactor;
    }
    baseColor.rgb *= fsIn.Color;

    // albedo
    vec3 albedo = baseColor.rgb;
    float alpha = (material.alphaMode == ALPHA_OPAQUE) ? 1.0 : baseColor.a;
    if (alpha < material.maskThreshold)
        discard;

    // metallic and roughness
    float metallic;
    float roughness;
    if (material.metalRoughnessCombined) {
        vec2 mr = texture(material.metallicMap, fsIn.TexCoords).rg;
        metallic = (!material.hasMetallicMap) ? material.metallic : mr.r;
        roughness = (!material.hasRoughnessMap) ? material.roughness : mr.g;
    }
    else {
        metallic = (!material.hasMetallicMap) ? material.metallic : texture(material.metallicMap, fsIn.TexCoords).r;
        roughness = (!material.hasRoughnessMap) ? material.roughness : texture(material.roughnessMap, fsIn.TexCoords).r;
    }
    metallic = material.metallicFactor * metallic;
    roughness = material.roughnessFactor * roughness;

    // input lighting data
    vec3 N = getNormal();
#ifdef VIEW_DEPENDENT_LIGHTING
    vec3 V = normalize(camera.position - fsIn.FragPos);
#else
    vec3 V = vec3(0.0, -1.0, 0.0); // Default view direction
#endif
    vec3 R = reflect(-V, N);

    // calculate reflectance at normal incidence; if dia-electric (like plastic) use F0
    // of 0.04 and if it's a metal, use the albedo baseColor as F0 (metallic workflow)
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);

    PBRInfo pbrInputs = PBRInfo(N, V, R, albedo, metallic, roughness, F0);

    // apply reflectance equation for lights
    vec3 radianceOut = vec3(0.0);
    radianceOut += calcDirLight(directionalLight, pbrInputs);
    for (int i = 0; i < numPointLights; i++) {
#ifdef PLATFORM_CORE
        radianceOut += calcPointLight(pointLights[i], pointLightShadowMaps[i], pbrInputs);
#else
        radianceOut += calcPointLight(pointLights[i], pbrInputs);
#endif
    }

    vec3 ambient = ambientLight.intensity * ambientLight.color * albedo;
    // apply IBL
    ambient += material.IBL * getIBLContribution(pbrInputs);

    // apply emissive component
    if (material.hasEmissiveMap) {
        vec3 emissive = texture(material.emissiveMap, fsIn.TexCoords).rgb;
        radianceOut += material.emissiveFactor * emissive;
    }

    // apply ambient occlusion
    if (material.hasAOMap) {
        float ao = texture(material.aoMap, fsIn.TexCoords).r;
        ambient *= ao;
    }

    radianceOut = radianceOut + ambient;

    FragColor = vec4(radianceOut, alpha);
    FragPosition = vec4(fsIn.FragPos, 1.0);
    FragNormal = vec4(normalize(fsIn.Normal), 1.0);
    FragIDs = uvec4(fsIn.drawID, gl_PrimitiveID, 0, 1);
    FragIDs.z = floatBitsToUint((-fsIn.FragPosView.z - camera.near) / (camera.far - camera.near));
}
