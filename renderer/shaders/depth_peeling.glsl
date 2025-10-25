#ifdef DO_DEPTH_PEELING
uniform bool peelDepth;
uniform usampler2D prevIDMap;

uniform int height;
uniform float E;
uniform float edpDelta;
uniform int layerIndex;

// Adapted from https://github.com/cgskku/pvhv/blob/main/shaders/edp.frag
#define DP_EPSILON 0.0005
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

    float blockerDepthNormalized = uintBitsToFloat(q.z);
    float df = mix(camera.near, camera.far, blockerDepthNormalized);
    float R = LCOC(fragmentDepth, df);
    for (int i = 0; i < EDP_SAMPLES; i++) {
        // Sample around a circle with radius R
        float x = R * cos(float(i) * 2*PI / EDP_SAMPLES);
        float y = R * sin(float(i) * 2*PI / EDP_SAMPLES);
        vec2 offset = vec2(x, y);

        uvec4 w = texelFetch(prevIDMap, ivec2(round(vec2(pixelCoords) + offset)), 0);

        float sampleDepthNormalized = uintBitsToFloat(w.z);
        if (sampleDepthNormalized == 0) return true;
        if (sampleDepthNormalized >= MAX_DEPTH) continue;

        int prevAlphaMode = int(w.w);
        if (prevAlphaMode != ALPHA_OPAQUE) return true;

        if (sampleDepthNormalized >= blockerDepthNormalized + edpDelta) return true;
        else if (sampleDepthNormalized <= blockerDepthNormalized - edpDelta) return true;
    }

    return false;
}

void applyDepthPeeling(vec3 fragViewPos) {
    if (peelDepth) {
        ivec2 pixelCoords = ivec2(gl_FragCoord.xy);
        uvec4 q = texelFetch(prevIDMap, pixelCoords, 0);

        float currDepth = -fragViewPos.z;
        float prevDepthNormalized = uintBitsToFloat(q.z);
        if (prevDepthNormalized == 0 || prevDepthNormalized >= MAX_DEPTH)
            discard;
        if (currDepth <= mix(camera.near, camera.far, prevDepthNormalized) + DP_EPSILON)
            discard;
#ifdef EDP
        int prevAlphaMode = int(q.w);
        if ((prevAlphaMode == ALPHA_OPAQUE) && !inPVHV(pixelCoords, fragViewPos, q))
            discard;
#endif
    }
}
#endif
