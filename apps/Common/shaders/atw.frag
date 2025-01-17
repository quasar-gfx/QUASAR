out vec4 FragColor;

in vec2 TexCoords;

#ifdef ANDROID
flat in float IsLeftEye;

uniform mat4 projectionInverseLeft;
uniform mat4 projectionInverseRight;
uniform mat4 viewInverseRight;
uniform mat4 viewInverseLeft;

uniform mat4 remoteProjectionLeft;
uniform mat4 remoteProjectionRight;
uniform mat4 remoteViewLeft;
uniform mat4 remoteViewRight;
#else
uniform mat4 projectionInverse;
uniform mat4 viewInverse;

uniform mat4 remoteProjection;
uniform mat4 remoteView;
#endif

uniform bool atwEnabled;
uniform bool toneMap;

#ifndef ANDROID
uniform bool toneMap = true;
uniform bool gammaCorrect = true;
uniform float exposure = 1.0;
#else
uniform bool toneMap;
uniform bool gammaCorrect;
uniform float exposure;
#endif

uniform sampler2D videoTexture;

const float epsilon = 0.0001;

vec3 linearToSRGB(vec3 color) {
    return mix(pow(color, vec3(1.0 / 2.4)) * 1.055 - 0.055, color * 12.92, lessThanEqual(color, vec3(0.0031308)));
}

vec3 ndcToView(mat4 projectionInverse, vec2 ndc, float depth) {
    vec4 ndcPos;
    ndcPos.xy = ndc;
    ndcPos.z = 2.0 * depth - 1.0;
    ndcPos.w = 1.0;

    vec4 viewCoord = projectionInverse * ndcPos;
    viewCoord = viewCoord / viewCoord.w;
    return viewCoord.xyz;
}

vec3 viewToWorld(mat4 viewInverse, vec3 viewCoord) {
    vec4 worldCoord = viewInverse * vec4(viewCoord, 1.0);
    worldCoord = worldCoord / worldCoord.w;
    return worldCoord.xyz;
}

vec3 worldToView(mat4 view, vec3 worldCoord) {
    vec4 viewCoord = view * vec4(worldCoord, 1.0);
    viewCoord = viewCoord / viewCoord.w;
    return viewCoord.xyz;
}

vec2 viewToNDC(mat4 projection, vec3 viewCoord) {
    vec4 ndcCoord = projection * vec4(viewCoord, 1.0);
    ndcCoord = ndcCoord / ndcCoord.w;
    return ndcCoord.xy;
}

vec2 worldToScreen(mat4 view, mat4 projection, vec3 worldCoord) {
    vec2 ndc = viewToNDC(projection, worldToView(view, worldCoord));
    vec2 uv = (ndc + 1.0) / 2.0;
    return uv;
}

void main() {
    vec2 TexCoordsAdjusted = TexCoords;
#ifdef ANDROID
    if (IsLeftEye > 0.5) {
        TexCoordsAdjusted.x = TexCoords.x / 2.0;
    }
    else {
        TexCoordsAdjusted.x = TexCoords.x / 2.0 + 0.5;
    }
#endif

    vec3 color;
    if (!atwEnabled) {
<<<<<<< HEAD
        color = texture(videoTexture, TexCoordsAdjusted).rgb;
=======
        vec3 color = texture(videoTexture, TexCoordsAdjusted).rgb;
        if (toneMap) color = linearToSRGB(color);
        FragColor = vec4(color, 1.0);
>>>>>>> 7376185f (fix atw)
        return;
    }
    else {
        vec2 ndc = TexCoords * 2.0 - 1.0;

#ifdef ANDROID
        vec3 viewCoord;
        vec3 worldCoord;
        vec2 TexCoordsRemote;
        if (IsLeftEye > 0.5) {
            viewCoord = ndcToView(projectionInverseLeft, ndc, 1.0);
            worldCoord = viewToWorld(mat4(mat3(viewInverseLeft)), viewCoord);
            TexCoordsRemote = worldToScreen(mat4(mat3(remoteViewLeft)), remoteProjectionLeft, worldCoord);
            TexCoordsRemote.x = clamp(TexCoordsRemote.x / 2.0, 0.0, 0.5 - epsilon);
        }
        else {
            viewCoord = ndcToView(projectionInverseRight, ndc, 1.0);
            worldCoord = viewToWorld(mat4(mat3(viewInverseRight)), viewCoord);
            TexCoordsRemote = worldToScreen(mat4(mat3(remoteViewRight)), remoteProjectionRight, worldCoord);
            TexCoordsRemote.x = clamp(TexCoordsRemote.x / 2.0 + 0.5, 0.5, 1.0 - epsilon);
        }
#else
        vec3 viewCoord = ndcToView(projectionInverse, ndc, 1.0);
        vec3 worldPose = viewToWorld(mat4(mat3(viewInverse)), viewCoord);
        vec2 TexCoordsRemote = worldToScreen(mat4(mat3(remoteView)), remoteProjection, worldPose);
#endif

<<<<<<< HEAD
        color = texture(videoTexture, TexCoordsRemote).rgb;
    }

    if (toneMap) {
        vec3 toneMappedResult = vec3(1.0) - exp(-color.rgb * exposure);
        color = toneMappedResult;
        if (gammaCorrect) {
            color = linearToSRGB(color);
        }
    }
=======
    vec3 color = texture(videoTexture, TexCoordsRemote).rgb;
    if (toneMap) color = linearToSRGB(color);
>>>>>>> 7376185f (fix atw)
    FragColor = vec4(color, 1.0);
}
