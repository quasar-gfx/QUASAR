out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D screenPositions;
uniform sampler2D screenNormals;
uniform sampler2D idBuffer;
uniform sampler2D screenColor;
uniform sampler2D screenDepth;

uniform sampler2D videoTexture;

uniform bool atwEnabled;

uniform mat4 projection;
uniform mat4 projectionInverse;
uniform mat4 viewInverse;

uniform mat4 remoteProjection;
uniform mat4 remoteView;

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

vec2 ndcToScreen(vec2 ndc) {
    return (ndc + 1.0) / 2.0;
}

vec2 worldToScreen(mat4 view, mat4 projection, vec3 worldCoord) {
    vec2 ndc = viewToNDC(projection, worldToView(view, worldCoord));
    return ndcToScreen(ndc);
}

void main() {
    if (!atwEnabled) {
        FragColor = vec4(texture(videoTexture, TexCoords).rgb, 1.0);
        return;
    }

    vec2 ndc = TexCoords * 2.0 - 1.0;
    vec3 viewCoord = ndcToView(projectionInverse, ndc, 1.0);
    vec3 worldPose = viewToWorld(viewInverse, viewCoord);
    vec2 TexCoordsRemote = worldToScreen(remoteView, remoteProjection, worldPose);

    FragColor = vec4(texture(videoTexture, TexCoordsRemote).rgb, 1.0);
}
