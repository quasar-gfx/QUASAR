#include "../constants.glsl"

layout(location = 0) out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D screenColor;
uniform sampler2D screenDepth;
uniform sampler2D screenNormals;
uniform sampler2D screenPositions;
uniform usampler2D idTexture;

uniform sampler2D noiseTexture;

uniform int kernelSize;
uniform float radius;
uniform float bias;

uniform vec3 samples[NUM_SAMPLES];

uniform mat4 view;
uniform mat4 projection;

void main() {
    vec3 fragPos = texture(screenPositions, TexCoord).xyz;
    vec3 normal  = normalize(texture(screenNormals,  TexCoord).xyz);

    vec2 texSize    = textureSize(screenNormals, 0);
    vec2 noiseScale = texSize / 4.0;
    vec3 randomVec  = normalize(texture(noiseTexture, TexCoord * noiseScale).xyz);
    vec3 tangent    = normalize(randomVec - normal * dot(randomVec, normal));
    vec3 bitangent  = cross(normal, tangent);
    mat3 TBN        = mat3(tangent, bitangent, normal);

    float occlusion = 0.0;
    for (int i = 0; i < kernelSize; ++i) {
        vec3 samplePosWS = fragPos + TBN * samples[i] * radius;

        vec4 clip = projection * view * vec4(samplePosWS, 1.0);
        clip.xyz /= max(clip.w, epsilon);
        vec2 sampleUV = clip.xy * 0.5 + 0.5;

        vec3 fetchedPosWS = texture(screenPositions, sampleUV).xyz;

        float samplePosVZ   = (view * vec4(samplePosWS, 1.0)).z;
        float fetchedPosVZ  = (view * vec4(fetchedPosWS, 1.0)).z;

        float range = smoothstep(0.0, 1.0, radius / abs((view * vec4(fragPos,1.0)).z - fetchedPosVZ));
        occlusion += (fetchedPosVZ > samplePosVZ + bias ? 1.0 : 0.0) * range;
    }

    occlusion = 1.0 - (occlusion / float(kernelSize));
    FragColor = vec4(occlusion, 0.0, 0.0, 1.0);
}
