out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D screenColor;
uniform sampler2D screenDepth;
uniform sampler2D screenNormals;
uniform sampler2D screenPositions;
uniform usampler2D idTexture;

uniform sampler2D noiseTexture;

int kernelSize = 64;
float radius = 0.5;
float bias = 0.025;

uniform vec3 samples[64];

uniform mat4 view;
uniform mat4 projection;

void main() {
    vec2 texSize = textureSize(screenNormals, 0);
    vec2 noiseScale = texSize / 4.0;

    vec3 fragPos = texture(screenPositions, TexCoord).xyz;
    // Extract the rotation part of the view matrix
    mat3 rotationMatrix = mat3(view);
    vec3 normal = normalize(rotationMatrix * texture(screenNormals, TexCoord).rgb);

    vec3 randomVec = normalize(texture(noiseTexture, TexCoord * noiseScale).xyz);

    vec3 tangent = normalize(randomVec - normal * dot(randomVec, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 TBN = mat3(tangent, bitangent, normal);

    float occlusion = 0.0;
    for (int i = 0; i < kernelSize; i++) {
        vec3 samplePos = fragPos + (TBN * samples[i]) * radius;

        vec4 offset = projection * vec4(samplePos, 1.0);
        offset.xyz /= offset.w;
        offset.xyz = offset.xyz * 0.5 + 0.5;

        float sampleDepth = texture(screenPositions, offset.xy).z;

        float rangeCheck = smoothstep(0.0, 1.0, radius / abs(fragPos.z - sampleDepth));
        occlusion += (sampleDepth >= samplePos.z + bias ? 1.0 : 0.0) * rangeCheck;
    }

    occlusion = 1.0 - (occlusion / kernelSize);
    FragColor = vec4(vec3(occlusion), 1.0);
}
