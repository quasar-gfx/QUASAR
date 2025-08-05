#include "../constants.glsl"

struct QuadMapData {
    vec3 normal;
    float depth;
    uvec2 offset;
    uint size;
    bool flattened;
};

struct QuadMapDataPacked {
    uint normalSpherical;
    float depth;
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
    return depth != 0.0 && depth < MAX_DEPTH;
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

uint packNormalToSpherical(vec3 normal) {
    // Convert to spherical coordinates
    float theta = acos(clamp(normal.y, -1.0, 1.0)); // elevation
    float phi = atan(normal.z, normal.x);           // azimuth

    float thetaSnorm = (theta / PI) * 2.0 - 1.0;
    float phiSnorm = phi / PI;

    // Pack into lower two components, pad upper two with 0
    return packSnorm4x8(vec4(thetaSnorm, phiSnorm, 0.0, 0.0));
}

vec3 unpackSphericalToNormal(uint packedNormal) {
    vec4 unpacked = unpackSnorm4x8(packedNormal);

    float thetaSnorm = unpacked.x;
    float phiSnorm = unpacked.y;

    float theta = (thetaSnorm + 1.0) * 0.5 * PI;
    float phi = phiSnorm * PI;

    // Reconstruct normal from spherical coords
    float y = cos(theta);
    float r = sin(theta);
    float x = r * cos(phi);
    float z = r * sin(phi);

    return normalize(vec3(x, y, z));
}
