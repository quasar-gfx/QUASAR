const float gamma = 2.4;

vec3 saturate(vec3 x) {
    return clamp(x, 0.0, 1.0);
}

vec3 linearToSRGB(vec3 color) {
    return mix(pow(color, vec3(1.0 / gamma)) * 1.055 - 0.055, color * 12.92, lessThanEqual(color, vec3(0.0031308)));
}

vec3 sRGBToLinear(vec3 color) {
    return mix(pow((color + 0.055) / 1.055, vec3(gamma)), color / 12.92, lessThanEqual(color, vec3(0.04045)));
}

vec3 tonemapReinhard(vec3 color, float exposure) {
    color = color * exposure;
    color = color / (color + vec3(1.0));
    return linearToSRGB(color);
}

vec3 tonemapExponential(vec3 color, float exposure) {
    color = color * exposure;
    color = vec3(1.0) - exp(-color);
    return linearToSRGB(color);
}

// See http://filmicworlds.com/blog/filmic-tonemapping-operators/
vec3 tonemapFilmic(vec3 color, float exposure) {
    vec3 temp = max(vec3(0.0), color * exposure - 0.004);
    return (temp * (vec3(6.2) * temp + vec3(0.5))) / (temp * (vec3(6.2) * temp + vec3(1.7)) + vec3(0.06));
}

// See https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
vec3 tonemapACESFilmic(vec3 color, float exposure) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    color = color * exposure;
    // See https://community.khronos.org/t/saturate/53155 for saturate impl
    return saturate((color * (a * color + b)) / (color * (c * color + d) + e));
}

// See http://filmicworlds.com/blog/filmic-tonemapping-operators/
vec3 tonemapUncharted2Impl(vec3 color) {
    const float a = 0.15;
    const float b = 0.50;
    const float c = 0.10;
    const float d = 0.20;
    const float e = 0.02;
    const float f = 0.30;
    return ((color * (a * color + c * b) + d * e) / (color * (a * color + b) + d * f)) - e / f;
}

vec3 tonemapUncharted2(vec3 color, float exposure) {
    const float W = 11.2;
    const float exposureBias = 2.0;
    color = tonemapUncharted2Impl(color * exposureBias * exposure);
    vec3 whiteScale = 1.0 / tonemapUncharted2Impl(vec3(W));
    return pow(color * whiteScale, vec3(1.0 / 2.2));
}

// Adapted from https://github.com/KhronosGroup/Tonemapping/blob/main/PBR_Neutral/pbrNeutral.glsl
vec3 tonemapKhronosPBR(vec3 color) {
    const float startCompression = 0.8 - 0.04;
    const float desaturation     = 0.15;

    float x    = min(color.x, min(color.y, color.z));
    float peak = max(color.x, max(color.y, color.z));

    float offset = x < 0.08 ? x * (-6.25 * x + 1.0) : 0.04;
    color -= offset;

    if(peak >= startCompression) {
        const float d       = 1.0 - startCompression;
        float       newPeak = 1.0 - d * d / (peak + d - startCompression);
        color *= newPeak / peak;

        float g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
        color   = mix(color, vec3(newPeak), vec3(g));
    }
    return linearToSRGB(color);
}
