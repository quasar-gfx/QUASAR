vec3 linearToSRGB(vec3 color) {
    return mix(pow(color, vec3(1.0 / 2.4)) * 1.055 - 0.055, color * 12.92, lessThanEqual(color, vec3(0.0031308)));
}

vec3 applyToneMapExponential(vec3 color, float exposure) {
    color = color * exposure;
    return vec3(1.0) - exp(-color);
}

vec3 applyToneMapReinhard(vec3 color, float exposure) {
    color = color * exposure;
    return color / (color + vec3(1.0));
}

vec3 saturate(vec3 x) {
    return clamp(x, 0.0, 1.0);
}

// See https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
vec3 applyToneMapACESFilm(vec3 color, float exposure) {
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    color = color * exposure;
    // See https://community.khronos.org/t/saturate/53155 for saturate impl
    return saturate((color * (a * color + b)) / (color * (c * color + d) + e));
}

// See http://filmicworlds.com/blog/filmic-tonemapping-operators/
vec3 applyToneMapUncharted2(vec3 color, float exposure) {
    const float A = 0.15;
    const float B = 0.50;
    const float C = 0.10;
    const float D = 0.20;
    const float E = 0.02;
    const float F = 0.30;
    color = color * exposure;
    return ((color*(A*color+C*B)+D*E)/(color*(A*color+B)+D*F))-E/F;
}
