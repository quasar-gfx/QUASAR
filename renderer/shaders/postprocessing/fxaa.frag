#include "tone_map.glsl"

out vec4 FragColor;
in vec2 TexCoord;

uniform sampler2D screenColor;

uniform bool toneMap = true;
uniform float exposure = 1.0;

// FXAA Parameters
uniform float contrastThreshold = 0.0625;
uniform float relativeThreshold = 0.125;
uniform float subpixelBlending = 1.0;

vec2 computeTexelSize(sampler2D tex) {
    return 1.0 / vec2(textureSize(tex, 0));
}

float computeLuminance(vec3 color) {
    return dot(color, vec3(0.299, 0.587, 0.114));
}

vec3 performToneMapping(vec3 color) {
    if (toneMap) {
        color = applyToneMapExponential(color, exposure);
        color = linearToSRGB(color);
    }
    return color;
}

// Adapted from: https://github.com/KTStephano/StratusGFX/blob/master/Source/Shaders/fxaa_smoothing.fs
void main() {
    vec3 color = texture(screenColor, TexCoord).rgb;
    float lumaCenter = computeLuminance(color);

    float lumaRight  = computeLuminance(textureOffset(screenColor, TexCoord, ivec2( 1,  0)).rgb);
    float lumaLeft   = computeLuminance(textureOffset(screenColor, TexCoord, ivec2(-1,  0)).rgb);
    float lumaTop    = computeLuminance(textureOffset(screenColor, TexCoord, ivec2( 0,  1)).rgb);
    float lumaBot    = computeLuminance(textureOffset(screenColor, TexCoord, ivec2( 0, -1)).rgb);

    float lumaMax = max(max(max(max(lumaCenter, lumaRight), lumaLeft), lumaTop), lumaBot);
    float lumaMin = min(min(min(min(lumaCenter, lumaRight), lumaLeft), lumaTop), lumaBot);

    float contrast = lumaMax - lumaMin;
    float threshold = max(contrastThreshold, relativeThreshold * lumaMax);

    if (contrast < threshold) {
        FragColor = vec4(performToneMapping(color), 1.0);
        return;
    }

    float lumaTopRight = computeLuminance(textureOffset(screenColor, TexCoord, ivec2( 1,  1)).rgb);
    float lumaBotRight = computeLuminance(textureOffset(screenColor, TexCoord, ivec2( 1, -1)).rgb);
    float lumaTopLeft  = computeLuminance(textureOffset(screenColor, TexCoord, ivec2(-1,  1)).rgb);
    float lumaBotLeft  = computeLuminance(textureOffset(screenColor, TexCoord, ivec2(-1, -1)).rgb);

    float average = (2.0 * (lumaRight + lumaLeft + lumaTop + lumaBot) +
                     lumaTopRight + lumaBotRight + lumaTopLeft + lumaBotLeft) / 12.0;

    float centerAvgContrast = abs(average - lumaCenter);
    float normalizedCenterAvgContrast = clamp(centerAvgContrast / contrast, 0.0, 1.0);

    float pixelBlendFactor = smoothstep(0.0, 1.0, normalizedCenterAvgContrast) *
                             smoothstep(0.0, 1.0, normalizedCenterAvgContrast) *
                             subpixelBlending;

    float horizontal = abs(lumaTop + lumaBot - 2.0 * lumaCenter) * 2.0 +
                       abs(lumaTopRight + lumaBotRight - 2.0 * lumaRight) +
                       abs(lumaTopLeft  + lumaBotLeft  - 2.0 * lumaLeft);

    float vertical   = abs(lumaRight + lumaLeft - 2.0 * lumaCenter) * 2.0 +
                       abs(lumaTopRight + lumaTopLeft - 2.0 * lumaTop) +
                       abs(lumaBotRight + lumaBotLeft - 2.0 * lumaBot);

    bool isHorizontal = horizontal >= vertical;
    vec2 texelSize = computeTexelSize(screenColor);
    float pixelStep = isHorizontal ? texelSize.y : texelSize.x;

    float positiveLuma = isHorizontal ? lumaTop : lumaRight;
    float negativeLuma = isHorizontal ? lumaBot : lumaLeft;
    float positiveGradient = abs(positiveLuma - lumaCenter);
    float negativeGradient = abs(negativeLuma - lumaCenter);

    float gradient;
    float oppositeLuma;
    if (positiveGradient < negativeGradient) {
        pixelStep = -pixelStep;
        oppositeLuma = negativeLuma;
        gradient = negativeGradient;
    }
    else {
        oppositeLuma = positiveLuma;
        gradient = positiveGradient;
    }

    vec2 uv = TexCoord;
    vec2 uvEdge = uv;
    vec2 edgeStep;

    if (isHorizontal) {
        uvEdge.y += pixelStep * 0.5;
        edgeStep = vec2(texelSize.x, 0.0);
    }
    else {
        uvEdge.x += pixelStep * 0.5;
        edgeStep = vec2(0.0, texelSize.y);
    }

    float edgeLumaAvg = (lumaCenter + oppositeLuma) * 0.5;
    float gradientThreshold = gradient * 0.25;

    bool atEdgeEnds[2] = bool[](false, false);
    float edgeToCenterDistances[2] = float[](0.0, 0.0);
    float edgeLumaDeltas[2] = float[](0.0, 0.0);
    float directionalSigns[2] = float[](1.0, -1.0);
    vec2 edgeSteps[2] = vec2[](edgeStep, -edgeStep);
    float edgeStepOffsets[10] = float[](1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0);
    float edgeGuessJumpSize = 8.0;

    for (int i = 0; i < 2; i++) {
        vec2 puv = uvEdge + edgeSteps[i] * edgeStepOffsets[0];
        float edgeLumaDelta = computeLuminance(texture(screenColor, puv).rgb) - edgeLumaAvg;
        bool atEdgeEnd = abs(edgeLumaDelta) >= gradientThreshold;

        for (int j = 0; j < 9 && !atEdgeEnd; ++j) {
            puv += edgeSteps[i] * edgeStepOffsets[j + 1];
            edgeLumaDelta = computeLuminance(texture(screenColor, puv).rgb) - edgeLumaAvg;
            atEdgeEnd = abs(edgeLumaDelta) >= gradientThreshold;
        }

        if (!atEdgeEnd) {
            puv += edgeSteps[i] * edgeGuessJumpSize;
        }

        edgeLumaDeltas[i] = edgeLumaDelta;
        atEdgeEnds[i] = atEdgeEnd;

        if (isHorizontal) {
            edgeToCenterDistances[i] = (directionalSigns[i] * puv.x) + (-directionalSigns[i] * uv.x);
        }
        else {
            edgeToCenterDistances[i] = (directionalSigns[i] * puv.y) + (-directionalSigns[i] * uv.y);
        }
    }

    float shortestDistance;
    bool deltaSign;
    if (edgeToCenterDistances[0] <= edgeToCenterDistances[1]) {
        shortestDistance = edgeToCenterDistances[0];
        deltaSign = edgeLumaDeltas[0] >= 0;
    }
    else {
        shortestDistance = edgeToCenterDistances[1];
        deltaSign = edgeLumaDeltas[1] >= 0;
    }

    float edgeBlendFactor;
    if (deltaSign == (lumaCenter - edgeLumaAvg >= 0)) {
        edgeBlendFactor = 0.0;
    }
    else {
        edgeBlendFactor = 0.5 - shortestDistance / (edgeToCenterDistances[0] + edgeToCenterDistances[1]);
    }

    float finalBlendFactor = max(pixelBlendFactor, edgeBlendFactor);

    if (isHorizontal) {
        uv.y += pixelStep * finalBlendFactor;
    }
    else {
        uv.x += pixelStep * finalBlendFactor;
    }

    color = texture(screenColor, uv).rgb;
    FragColor = vec4(performToneMapping(color), 1.0);
}
