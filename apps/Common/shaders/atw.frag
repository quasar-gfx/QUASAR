#include "camera_utils.glsl"

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

uniform sampler2D videoTexture;

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
        color = texture(videoTexture, TexCoordsAdjusted).rgb;
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

        color = texture(videoTexture, TexCoordsRemote).rgb;
    }

    FragColor = vec4(color, 1.0);
}
