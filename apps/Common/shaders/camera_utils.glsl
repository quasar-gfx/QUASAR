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

vec3 viewToNDC3(mat4 projection, vec3 viewCoord) {
    vec4 ndcCoord = projection * vec4(viewCoord, 1.0);
    ndcCoord = ndcCoord / ndcCoord.w;
    ndcCoord.z = (ndcCoord.z + 1.0) / 2.0;
    return ndcCoord.xyz;
}

vec2 viewToScreen(mat4 projection, vec3 viewCoord) {
    vec2 ndc = viewToNDC(projection, viewCoord).xy;
    vec2 uv = (ndc + 1.0) / 2.0;
    return uv;
}

vec2 worldToScreen(mat4 view, mat4 projection, vec3 worldCoord) {
    vec2 ndc = viewToNDC(projection, worldToView(view, worldCoord));
    vec2 uv = (ndc + 1.0) / 2.0;
    return uv;
}
