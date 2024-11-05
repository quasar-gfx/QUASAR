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
#endif

void main() {
#ifdef DO_DEPTH_PEELING
    if (peelDepth) {
        float depth = gl_FragCoord.z;
        vec2 screenCoords = gl_FragCoord.xy / vec2(textureSize(prevDepthMap, 0));
        float prevDepth = texture(prevDepthMap, screenCoords).r;
        // if the current fragment is closer than the previous fragment, discard it
        if (depth <= prevDepth)
            discard;
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
