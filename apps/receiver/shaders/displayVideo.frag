#version 410 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D screenPositions;
uniform sampler2D screenNormals;
uniform sampler2D screenColor;
uniform sampler2D screenDepth;

uniform sampler2D videoTexture;

uniform bool atwEnabled;

uniform mat4 projection;
uniform mat4 view;

uniform mat4 remoteProjection;
uniform mat4 remoteView;

vec3 cameraToWorld(vec2 screenPos, mat4 invProj, mat4 invView) {
    vec4 clipPos = vec4(screenPos * 2.0 - 1.0, 1.0, 1.0);
    vec4 worldPos = invView * invProj * clipPos;
    return worldPos.xyz / worldPos.w;
}

vec2 worldToCamera(vec3 worldPos, mat4 proj, mat4 view) {
    vec4 clipPos = proj * view * vec4(worldPos, 1.0);
    return clipPos.xy / clipPos.w * 0.5 + 0.5;
}

void main() {
    if (!atwEnabled) {
        FragColor = vec4(texture(videoTexture, TexCoords).rgb, 1.0);
        return;
    }

    vec3 worldPose = cameraToWorld(TexCoords, inverse(projection), inverse(view));
    vec2 TexCoordsRemote = worldToCamera(worldPose, remoteProjection, remoteView);

    FragColor = vec4(texture(videoTexture, TexCoordsRemote).rgb, 1.0);
}
