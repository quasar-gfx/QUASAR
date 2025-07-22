#include "constants.glsl"
#include "camera.glsl"
#include "pbr.glsl"

layout(location = 0) out vec4 FragColor;
layout(location = 1) out vec4 FragNormal;
layout(location = 2) out uvec4 FragIDs;

in VertexData {
    flat uint drawID;
    vec2 TexCoords;
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

uniform AmbientLight ambientLight;
uniform DirectionalLight directionalLight;
layout(std140) uniform PointLightBlock {
    PointLight pointLights[MAX_POINT_LIGHTS];
    int numPointLights;
}

// Shadow maps
uniform sampler2D dirLightShadowMap; // 9
uniform samplerCube pointLightShadowMaps[MAX_POINT_LIGHTS]; // 10+

#ifdef DO_DEPTH_PEELING
uniform bool peelDepth;
uniform usampler2D prevIDMap;

uniform int height;
uniform float E;
uniform float edpDelta;
uniform int layerIndex;
#endif

vec3 getNormal() {
	vec3 N = normalize(fsIn.Normal);
	vec3 T = normalize(fsIn.Tangent);
	vec3 B = normalize(fsIn.BiTangent);

    if (!material.hasNormalMap)
        return N;

    if (any(isnan(B))) {
        vec3 q1 = dFdx(fsIn.FragPosWorld);
        vec3 q2 = dFdy(fsIn.FragPosWorld);
        vec2 st1 = dFdx(fsIn.TexCoords);
        vec2 st2 = dFdy(fsIn.TexCoords);

        T = normalize(q1 * st2.t - q2 * st1.t);
        B = -normalize(cross(N, T));
    }

	mat3 TBN = mat3(T, B, N);
	vec3 tangentNormal = texture(material.normalMap, fsIn.TexCoords).xyz * 2.0 - 1.0;
	return normalize(TBN * tangentNormal);
}

#ifdef DO_DEPTH_PEELING

// Adapted from https://github.com/cgskku/pvhv/blob/main/shaders/edp.frag
#define DP_EPSILON 0.0001
#define EDP_SAMPLES 16

bool cullUmbra(float fragmentDepth, float zf) {
    float d = fragmentDepth; // fragment depth
	float df = mix(camera.near, camera.far, zf); // blocker depth
	float s  = tan(camera.fovy * 0.5) * 2.0 * df / height; // pixel geometry size
	if (E < s) return true; // no more peeling, because the pixel geometry size > lens size
	float x  = df * s / (E - s);
	return d < df + x;
}

float LCOC(float d, float df) {
	float K = float(height)*0.5 / df / tan(camera.fovy*0.5); // screen-space LCOC scale
	return K * E * abs(df-d) / d; // relative radius of COC against df (blocker depth)
}

bool inPVHV(ivec2 pixelCoords, vec3 fragViewPos, uvec4 q) {
    float fragmentDepth = -fragViewPos.z;

    if (layerIndex > 2) return cullUmbra(fragmentDepth, uintBitsToFloat(q.z));

    uint q_item = q.r;
    if (q_item < 0) return false;

    float blockerDepthNormalized = uintBitsToFloat(q.z);
	float df = mix(camera.near, camera.far, blockerDepthNormalized);
    float R = LCOC(fragmentDepth, df);
    for (int i = 0; i < EDP_SAMPLES; i++) {
        // Sample around a circle with radius R
        float x = R * cos(float(i) * 2*PI / EDP_SAMPLES);
        float y = R * sin(float(i) * 2*PI / EDP_SAMPLES);
        vec2 offset = vec2(x, y);

        uvec4 w = texelFetch(prevIDMap, ivec2(round(vec2(pixelCoords) + offset)), 0);
        uint w_item = w.r;
        if (w_item < 0) return false;

        float sampleDepthNormalized = uintBitsToFloat(w.z);
        if (sampleDepthNormalized == 0) return true;
        if (sampleDepthNormalized >= MAX_DEPTH) continue;

        if (sampleDepthNormalized >= blockerDepthNormalized + edpDelta) return true;
        else if (sampleDepthNormalized <= blockerDepthNormalized - edpDelta) return true;
    }

    return false;
}
#endif

void main() {
#ifdef DO_DEPTH_PEELING
    if (peelDepth) {
        ivec2 pixelCoords = ivec2(gl_FragCoord.xy);
        uvec4 q = texelFetch(prevIDMap, pixelCoords, 0);

        float currDepth = -fsIn.FragPosView.z;
        float prevDepthNormalized = uintBitsToFloat(q.z);
        if (prevDepthNormalized == 0 || prevDepthNormalized >= MAX_DEPTH)
            discard;
        if (-fsIn.FragPosView.z <= mix(camera.near, camera.far, prevDepthNormalized + DP_EPSILON))
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

    // Albedo
    vec3 albedo = baseColor.rgb;
    float alpha = (material.alphaMode == ALPHA_OPAQUE) ? 1.0 : baseColor.a;
    if (alpha < material.maskThreshold)
        discard;

    // Metallic and roughness properties
    float metallic, roughness;
    if (material.metalRoughnessCombined) {
        vec4 mrSample = texture(material.metallicMap, fsIn.TexCoords);
        metallic = (!material.hasMetallicMap) ? material.metallic : mrSample.b;
        roughness = (!material.hasRoughnessMap) ? material.roughness : mrSample.g;
    }
    else {
        metallic = (!material.hasMetallicMap) ? material.metallic : texture(material.metallicMap, fsIn.TexCoords).r;
        roughness = (!material.hasRoughnessMap) ? material.roughness : texture(material.roughnessMap, fsIn.TexCoords).r;
    }
    metallic = material.metallicFactor * metallic;
    roughness = material.roughnessFactor * roughness;

    // Input lighting data
    vec3 N = getNormal();
    vec3 V = normalize(camera.position - fsIn.FragPosWorld);
    vec3 R = reflect(-V, N);

    // Calculate reflectance at normal incidence; if dia-electric (like plastic) use F0
    // Of 0.04 and if it's a metal, use the albedo baseColor as F0 (metallic workflow)
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);

    PBRInfo pbrInputs = PBRInfo(N, V, R, albedo, metallic, roughness, F0);

    // Apply reflectance equation for lights
    vec3 radianceOut = vec3(0.0);
    radianceOut += calcDirLight(directionalLight, pbrInputs, dirLightShadowMap, fsIn.FragPosLightSpace, fsIn.Normal);
    for (int i = 0; i < numPointLights; i++) {
        PointLight light = pointLights[i];
        radianceOut += calcPointLight(light, pointLightShadowMaps[light.shadowIndex], pbrInputs, fsIn.FragPosWorld);
    }

    vec3 ambient = ambientLight.intensity * ambientLight.color * albedo;
    // Apply IBL
    ambient += material.IBL * calcIBLContribution(pbrInputs, material.irradianceMap, material.prefilterMap, material.brdfLUT);

    // Apply emissive component
    if (material.hasEmissiveMap) {
        vec3 emissive = texture(material.emissiveMap, fsIn.TexCoords).rgb;
        radianceOut += material.emissiveFactor * emissive;
    }

    // Apply ambient occlusion
    if (material.hasAOMap) {
        float ao = texture(material.aoMap, fsIn.TexCoords).r;
        ambient *= ao;
    }

    radianceOut = radianceOut + ambient;

    FragColor = vec4(radianceOut, alpha);
    FragNormal = vec4(normalize(fsIn.Normal), 1.0);
    FragIDs = uvec4(fsIn.drawID, gl_PrimitiveID, 0, 1);
    FragIDs.z = floatBitsToUint((-fsIn.FragPosView.z - camera.near) / (camera.far - camera.near));
}
