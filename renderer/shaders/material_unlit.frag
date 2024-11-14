layout(location = 0) out vec4 FragColor;
layout(location = 1) out vec4 FragPosition;
layout(location = 2) out vec4 FragNormal;
layout(location = 3) out vec4 FragIDs;

in VertexData {
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

    bool hasBaseColorMap; // use diffuse map

    // material textures
    sampler2D baseColorMap; // 0
} material;

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
uniform sampler2D prevDepthMap;

uniform int height;
uniform float E;
uniform float edpDelta;

const float PI = 3.1415926535897932384626433832795;

#define MAX_DEPTH 0.9999

#define EDP_SAMPLES 16

float linearizeAndNormalizeDepth(float depth) {
    float z = depth * 2.0 - 1.0; // back to NDC
    float linearized = (2.0 * camera.near * camera.far) / (camera.far + camera.near - z * (camera.far - camera.near));
    return (linearized - camera.near) / (camera.far - camera.near);
}

float LCOC(float d, float df) {
	float K = float(height)*0.5 / df / tan(camera.fovy*0.5); // screen-space LCOC scale
	return K * E * abs(df-d) / d; // relative radius of COC against df (blocker depth)
}

bool inPVHV(ivec2 pixelCoords, vec3 fragViewPos, float blockerDepthNonLinear) {
    float fragmentDepth = -fragViewPos.z;
    float blockerDepthNormalized = linearizeAndNormalizeDepth(blockerDepthNonLinear);

	float df = mix(camera.near, camera.far, blockerDepthNormalized);
    float R = LCOC(fragmentDepth, df);
    for (int i = 0; i < EDP_SAMPLES; i++) {
        // sample around a circle with radius R
        float x = R * cos(float(i) * 2*PI / EDP_SAMPLES);
        float y = R * sin(float(i) * 2*PI / EDP_SAMPLES);
        vec2 offset = vec2(x, y);

        float sampleDepthNonLinear = texelFetch(prevDepthMap, ivec2(round(vec2(pixelCoords) + offset)), 0).r;
        float sampleDepthNormalized = linearizeAndNormalizeDepth(sampleDepthNonLinear);
        if (sampleDepthNormalized == 0) return true;
        if (sampleDepthNormalized >= MAX_DEPTH) continue;

        if      (sampleDepthNormalized >= blockerDepthNormalized + edpDelta) return true;
        else if (sampleDepthNormalized <= blockerDepthNormalized - edpDelta) return true;
    }
    return false;
}
#endif

void main() {
#ifdef DO_DEPTH_PEELING
    if (peelDepth) {
        ivec2 pixelCoords = ivec2(gl_FragCoord.xy);
        float currDepth = gl_FragCoord.z;
        float prevDepth = texelFetch(prevDepthMap, pixelCoords, 0).r;
        if (currDepth <= prevDepth)
            discard;
#ifdef EDP
        vec3 fragViewPos = fsIn.FragPosView;
        if (!inPVHV(pixelCoords, fragViewPos, prevDepth))
            discard;
#endif
    }
#endif

    vec4 baseColor;
    if (material.hasBaseColorMap) {
        baseColor = texture(material.baseColorMap, fsIn.TexCoords) * material.baseColorFactor;
    }
    else {
        baseColor = material.baseColor * material.baseColorFactor;
    }
    baseColor.rgb *= fsIn.Color;

    float alpha = (material.alphaMode == ALPHA_OPAQUE) ? 1.0 : baseColor.a;
    if (alpha < material.maskThreshold)
        discard;

    FragColor = vec4(baseColor.rgb, alpha);
    FragPosition = vec4(fsIn.FragPos, 1.0);
    FragNormal = vec4(normalize(fsIn.Normal), 1.0);
    FragIDs = vec4(gl_PrimitiveID, 0.0, 0.0, 0.0);
}
