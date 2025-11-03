#include "../constants.glsl"

struct QuadMapData {
    vec3 normal;
    float depth;
    uvec2 offset;
    uint size;
    bool hasAlpha;
    bool flattened;
};

struct QuadMapDataPacked {
    uint normalAndDepth;
    uint metadata;
};

struct Quad {
    vec3 bottomLeft;
    vec3 bottomRight;
    vec3 topLeft;
    vec3 topRight;
};

struct Quad2D {
    vec2 bottomLeft;
    vec2 bottomRight;
    vec2 topLeft;
    vec2 topRight;
};

// A Quad subdivided into 4 smaller Quads
struct MultiQuad {
    vec3 bottomLeft;
    vec3 bottom;
    vec3 bottomRight;
    vec3 left;
    vec3 center;
    vec3 right;
    vec3 topLeft;
    vec3 top;
    vec3 topRight;
};

struct MultiQuad1D {
    float bottomLeft;
    float bottom;
    float bottomRight;
    float left;
    float center;
    float right;
    float topLeft;
    float top;
    float topRight;
};

struct MultiQuad2D {
    vec2 bottomLeft;
    vec2 bottom;
    vec2 bottomRight;
    vec2 left;
    vec2 center;
    vec2 right;
    vec2 topLeft;
    vec2 top;
    vec2 topRight;
};

struct Plane {
    vec3 normal;
    vec3 point;
};

const vec3 origin = vec3(0.0, 0.0, 0.0);
const vec3 forward = vec3(0.0, 0.0, -1.0);
const vec3 up = vec3(0.0, 1.0, 0.0);
const vec3 left = vec3(-1.0, 0.0, 0.0);
const vec3 right = vec3(1.0, 0.0, 0.0);

const float surfelSize = 0.5;

bool isValidDepth(float depth) {
    return depth != 0.0 && depth < 1.0;
}

bool isValidQuadMapData(in QuadMapData quadMapData) {
    return quadMapData.normal != vec3(0.0) &&
           isValidDepth(quadMapData.depth) &&
           quadMapData.size > 0u;
}

float normalizeDepth(float depth, float near, float far) {
    return (depth - near) / (far - near);
}

float invDepthDistance(vec3 point1, vec3 point2, float near, float far) {
    float z1 = normalizeDepth(max(point1.z, near), near, far);
    float z2 = normalizeDepth(max(point2.z, near), near, far);
    return abs((1.0 / z1) - (1.0 / z2));
}

float signedDistance(vec3 p1, vec3 p2) {
    float dist = distance(p1, p2);
    return (length(p1) > length(p2)) ? -dist : dist;
}

vec3 rayPlaneIntersection(vec3 rayOrigin, vec3 rayDirection, Plane plane) {
    float denominator = dot(rayDirection, plane.normal);
    if (abs(denominator) < epsilon) {
        return vec3(1.0/0.0);
    }

    float t = dot(plane.point - rayOrigin, plane.normal) / denominator;
    if (t < 0.0) {
        return vec3(1.0/0.0);
    }

    vec3 intersection = rayOrigin + t * rayDirection;
    return intersection;
}

vec3 pointPlaneIntersection(vec3 pt, Plane plane) {
    vec3 rayDirection = normalize(pt);
    return rayPlaneIntersection(origin, rayDirection, plane);
}

// Adapted from: https://knarkowicz.wordpress.com/2014/04/16/octahedron-normal-vector-encoding/
vec2 warpOct(vec2 v) {
    float ox = (1.0 - abs(v.y)) * (v.x >= 0.0 ? 1.0 : -1.0);
    float oy = (1.0 - abs(v.x)) * (v.y >= 0.0 ? 1.0 : -1.0);
    return vec2(ox, oy);
}

uint packNormalToOctahedral(vec3 n) {
    // Project to octahedron
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    // Fold the bottom half onto the top
    if (n.z < 0.0) {
        vec2 o = warpOct(n.xy);
        n = vec3(o, n.z);
    }
    return packSnorm4x8(vec4(n.x, n.y, 0.0, 0.0));
}

vec3 unpackOctahedralToNormal(uint packedNormal) {
    vec2 f = unpackSnorm4x8(packedNormal).xy;

    // Reconstruct z and unfold if needed
    vec3 n = vec3(f.xy, 1.0 - abs(f.x) - abs(f.y));
    if (n.z < 0.0) {
        vec2 o = warpOct(n.xy);
        n = vec3(o, n.z);
    }
    return normalize(n);
}

uint packDepthUNORM16(float depth) {
    depth = clamp(depth, 0.0, 1.0);
    return uint(round(depth * 65535.0));
}

float unpackDepthUNORM16(uint bits) {
    return float(bits & 0xFFFFu) / 65535.0;
}
