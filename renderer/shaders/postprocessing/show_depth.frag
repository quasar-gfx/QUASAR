out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D screenColor;
uniform sampler2D screenDepth;
uniform sampler2D screenNormals;
uniform sampler2D screenPositions;
uniform usampler2D idTexture;

uniform float near;
uniform float far;

uniform float depthMultiplier;

float linearizeDepth01(float nonlinearDepth){
    float z = nonlinearDepth * 2.0 - 1.0;
    float viewZ = (2.0 * near * far) / (far + near - z * (far - near));
    return (viewZ - near) / (far - near);
}

void main() {
    float nonlinearDepth = texture(screenDepth, TexCoord).r;
    float depth = linearizeDepth01(nonlinearDepth) * depthMultiplier;
    FragColor = vec4(vec3(depth), 1.0);
}
