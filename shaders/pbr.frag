layout(location = 0) out vec4 positionBuffer;
layout(location = 1) out vec4 normalsBuffer;
layout(location = 2) out vec4 idBuffer;
layout(location = 3) out vec4 FragColor;

in VertexData {
    flat uint VertexID;
    vec2 TexCoords;
    vec3 FragPos;
    vec3 Color;
    vec3 Normal;
    vec3 Tangent;
    vec3 BiTangent;
    vec4 FragPosLightSpace;
} fsIn;

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

uniform int numPointLights;

uniform AmbientLight ambientLight;
uniform DirectionalLight directionalLight;
uniform PointLight pointLights[MAX_POINT_LIGHTS];

// material textures
uniform sampler2D albedoMap; // 0
uniform sampler2D normalMap; // 1
uniform sampler2D metallicMap; // 2
uniform sampler2D roughnessMap; // 3
uniform sampler2D aoMap; // 4

// IBL
uniform float IBL; // IBL contribution
uniform samplerCube irradianceMap; // 5
uniform samplerCube prefilterMap; // 6
uniform sampler2D brdfLUT; // 7

// shadow maps
uniform sampler2D dirLightShadowMap; // 8
uniform samplerCube pointLightShadowMaps[MAX_POINT_LIGHTS]; // 9+

uniform bool aoMapped;
uniform bool normalMapped;
uniform bool metalRoughnessCombined;

uniform vec3 baseColor;
uniform float opacity;
uniform bool transparent;

uniform float u_metallic;
uniform float u_roughness;

uniform vec3 camPos;

const float PI = 3.1415926535897932384626433832795;

vec3 gridSamplingDisk[20] = vec3[]
(
   vec3( 1,  1,  1), vec3( 1, -1,  1), vec3(-1, -1,  1), vec3(-1,  1,  1),
   vec3( 1,  1, -1), vec3( 1, -1, -1), vec3(-1, -1, -1), vec3(-1,  1, -1),
   vec3( 1,  1,  0), vec3( 1, -1,  0), vec3(-1, -1,  0), vec3(-1,  1,  0),
   vec3( 1,  0,  1), vec3(-1,  0,  1), vec3( 1,  0, -1), vec3(-1,  0, -1),
   vec3( 0,  1,  1), vec3( 0, -1,  1), vec3( 0, -1, -1), vec3( 0,  1, -1)
);

vec3 getNormalFromMap() {
    vec3 N   = normalize(fsIn.Normal);
    vec3 T   = normalize(fsIn.Tangent);
    vec3 B   = normalize(fsIn.BiTangent);
    mat3 TBN = mat3(T, B, N);

    if (!normalMapped) {
        return N;
    }

    vec3 normal = normalize(2.0 * texture(normalMap, fsIn.TexCoords).rgb - 1.0);
    return normalize(TBN * normal);
}

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

float calcDirLightShadow(DirectionalLight light, vec4 fragPosLightSpace) {
    // perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;

    // PCF
    int samples = 9;
    float shadow = 0.0;
    vec3 normal = normalize(fsIn.Normal);
    vec3 lightDir = normalize(-light.direction);
    float bias = max(0.05 * (1.0 - dot(normal, lightDir)), 0.005);
    vec2 texelSize = 1.0 / textureSize(dirLightShadowMap, 0);
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
    float viewDistance = length(camPos - fsIn.FragPos);
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

vec3 calcDirLight(DirectionalLight light, vec3 N, vec3 V, vec3 albedo, float roughness, float metallic, vec3 F0) {
    if (light.intensity == 0.0)
        return vec3(0.0);

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

vec3 calcPointLight(PointLight light, samplerCube pointLightShadowMap, vec3 N, vec3 V, vec3 albedo, float roughness, float metallic, vec3 F0) {
    if (light.intensity == 0.0)
        return vec3(0.0);

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
    float shadow = calcPointLightShadows(light, pointLightShadowMap, fragToLight);
    radianceOut *= (1.0 - shadow);

    return radianceOut;
}

void main() {
    // material properties
    vec4 color = texture(albedoMap, fsIn.TexCoords);
    if (color.rgb == vec3(0.0) && baseColor != vec3(-1.0)) {
        color.rgb = baseColor;
        color.a = opacity;
    }
    else {
        color.rgb *= fsIn.Color;
    }

    // albedo
    vec3 albedo = color.rgb;
    float alpha = (transparent) ? color.a : 1.0;
    if (alpha < 0.1)
        discard;

    // metallic and roughness
    vec2 mr = texture(metallicMap, fsIn.TexCoords).rg;
    float metallic = (u_metallic != -1.0) ? u_metallic : mr.r;
    float roughness = (u_roughness != -1.0) ? u_roughness : mr.g;
    if (!metalRoughnessCombined) {
        metallic = texture(metallicMap, fsIn.TexCoords).r;
        roughness = texture(roughnessMap, fsIn.TexCoords).r;
    }

    // ambient occlusion
    float ao = texture(aoMap, fsIn.TexCoords).r;

    // input lighting data
    vec3 N = getNormalFromMap();
    vec3 V = normalize(camPos - fsIn.FragPos);
    vec3 R = reflect(-V, N);

    // calculate reflectance at normal incidence; if dia-electric (like plastic) use F0
    // of 0.04 and if it's a metal, use the albedo color as F0 (metallic workflow)
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);

    // reflectance equation
    vec3 radianceOut = vec3(0.0);
    radianceOut += calcDirLight(directionalLight, N, V, albedo, roughness, metallic, F0);
    for (int i = 0; i < numPointLights; i++) {
        radianceOut += calcPointLight(pointLights[i], pointLightShadowMaps[i], N, V, albedo, roughness, metallic, F0);
    }

    // IBL
    vec3 ambient = ambientLight.intensity * ambientLight.color * albedo;
    if (IBL != 0.0) {
        // ambient lighting (we now use IBL as the ambient term)
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
        vec3 specular = prefilteredColor * (kS * brdf.x + brdf.y);
        ambient += IBL * (kD * diffuse + specular);
    }

    if (aoMapped) {
        ambient *= ao;
    }

    radianceOut = radianceOut + ambient;

    positionBuffer = vec4(fsIn.FragPos, 1.0);
    normalsBuffer = vec4(normalize(fsIn.Normal), 1.0);
    idBuffer = vec4(gl_PrimitiveID, 0.0, 0.0, 0.0);
    FragColor = vec4(radianceOut, alpha);
}
