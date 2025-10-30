#include "constants.glsl"
#include "tonemap.glsl"

layout(location = 0) out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D screenColor;
uniform sampler2D screenDepth;
uniform sampler2D screenNormals;
uniform sampler2D screenPositions;

uniform float depthThreshold;
uniform int searchRadius = 2;

uniform bool tonemap = true;
uniform float exposure = 1.0;

const int   MIN_DIRS_HIT     = 3;     // need geometry in >=3 directions -> avoids sky
const int   MIN_VALID_NEIGH  = 5;     // minimum neighbors to consider a fill
const float MAX_PLANE_RMS    = 0.01;  // meters (or your position units) - planarity check
const float MAX_N_ANGLE_COS  = 0.85;  // normals must broadly agree with the plane normal
const float MIN_TOTAL_WEIGHT = 0.10;  // reject weak fills

void fitPlane(in vec3 centroidPt, in vec3 avgNormal, out vec3 n, out float d) {
    n = normalize(avgNormal);
    d = -dot(n, centroidPt);
}

void main() {
    vec3  color       = texture(screenColor, TexCoord).rgb;
    float centerDepth = texture(screenDepth, TexCoord).r;

    // Only fill "holes"
    if (centerDepth >= MAX_DEPTH) {
        ivec2 dim = textureSize(screenDepth, 0);
        vec2 texel = 1.0 / vec2(dim);

        vec2 dirs[8] = vec2[8](
            vec2(+1, +0), vec2(-1, +0), vec2(+0, +1), vec2(+0, -1),
            vec2(+1, +1), vec2(-1, +1), vec2(+1, -1), vec2(-1, -1)
        );

        // Gather first finite hit in each direction
        bool hitFound[8]; for (int i = 0; i < 8; i++) hitFound[i] = false;
        int dirsHit = 0;

        for (int d = 0; d < 8; d++) {
            for (int step = 1; step <= searchRadius; step++) {
                vec2 uv = TexCoord + dirs[d] * (texel * float(step));
                float z = texture(screenDepth, uv).r;
                if (z < MAX_DEPTH) {
                    vec3 nrm = texture(screenNormals, uv).xyz;
                    if (!all(equal(nrm, nrm))) continue; // skip NaN normals (sky)
                    hitFound[d] = true;
                    dirsHit++;
                    break;
                }
            }
        }

        if (dirsHit >= MIN_DIRS_HIT) {
            // Build neighbor set within small kernel
            vec3 centroidPt = vec3(0.0);
            vec3 avgN = vec3(0.0);
            int count = 0;

            for (int x = -searchRadius; x <= searchRadius; ++x) {
                for (int y = -searchRadius; y <= searchRadius; ++y) {
                    if (x == 0 && y == 0) continue;
                    vec2 uv = TexCoord + vec2(x, y) * texel;

                    float z = texture(screenDepth, uv).r;
                    if (z >= MAX_DEPTH) continue;

                    vec3 nrm = texture(screenNormals, uv).xyz;
                    if (!all(equal(nrm, nrm))) continue; // skip NaN (sky)

                    vec3 pos = texture(screenPositions, uv).xyz;
                    centroidPt += pos;
                    avgN += nrm;
                    count++;
                }
            }

            if (count >= MIN_VALID_NEIGH) {
                centroidPt /= float(count);
                avgN = normalize(avgN);

                // Plane from (centroidPt, avg normal)
                vec3 n; float d;
                fitPlane(centroidPt, avgN, n, d);

                // Check planarity: RMS distance of neighbors to the plane
                float sumSq = 0.0;
                int used = 0;

                for (int x = -searchRadius; x <= searchRadius; x++) {
                    for (int y = -searchRadius; y <= searchRadius; y++) {
                        if (x == 0 && y == 0) continue;
                        vec2 uv = TexCoord + vec2(x, y) * texel;

                        float z = texture(screenDepth, uv).r;
                        if (z >= MAX_DEPTH) continue;

                        vec3 nrm = texture(screenNormals, uv).xyz;
                        if (!all(equal(nrm, nrm))) continue; // skip NaN (sky)

                        vec3 pos = texture(screenPositions, uv).xyz;
                        float dist = abs(dot(n, pos) + d);
                        sumSq += dist * dist;
                        used++;
                    }
                }

                float rms = sqrt(max(sumSq, 0.0) / max(1.0, float(used)));

                // If neighborhood is planar enough, fill using inverse distance weights,
                // while also checking that neighbor normals broadly align with the plane
                if (rms <= MAX_PLANE_RMS) {
                    vec3 sumColor = vec3(0.0);
                    float sumWeight = 0.0;

                    for (int x = -searchRadius; x <= searchRadius; x++) {
                        for (int y = -searchRadius; y <= searchRadius; y++) {
                            if (x == 0 && y == 0) continue;
                            vec2 uv = TexCoord + vec2(x, y) * texel;

                            float z = texture(screenDepth, uv).r;
                            if (z >= MAX_DEPTH) continue;

                            vec3 nS = texture(screenNormals, uv).xyz;
                            if (!all(equal(nS, nS))) continue; // skip NaN (sky)

                            vec3 pos = texture(screenPositions, uv).xyz;
                            float distPlane = abs(dot(n, pos) + d);
                            float nCos = clamp(dot(n, normalize(nS)), 0.0, 1.0);
                            if (nCos < MAX_N_ANGLE_COS) continue; // avoid corners/creases

                            float pixelDist = length(vec2(x, y));
                            float w = 1.0 / (1.0 + pixelDist) * 1.0 / (1.0 + distPlane * 100.0);
                            // (100.0 scales plane distance into pixel-ish units)

                            sumColor += texture(screenColor, uv).rgb * w;
                            sumWeight += w;
                        }
                    }

                    if (sumWeight > MIN_TOTAL_WEIGHT) {
                        color = sumColor / sumWeight;
                    }
                }
            }
        }
    }

    if (tonemap) {
        color = tonemapExponential(color, exposure);
    }

    FragColor = vec4(color, 1.0);
}
