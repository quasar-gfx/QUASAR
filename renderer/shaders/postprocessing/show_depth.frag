out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D screenColor;
uniform sampler2D screenDepth;
uniform sampler2D screenNormals;
uniform sampler2D screenPositions;
uniform usampler2D idTexture;

uniform float near = 0.1;
uniform float far = 1000.0;

float LinearizeDepth(float depth) {
    float z = depth * 2.0 - 1.0; // back to NDC
    return (2.0 * near * far) / (far + near - z * (far - near));
}

void main() {
    float depth = LinearizeDepth(texture(screenDepth, TexCoord).r) / (far - near);
    FragColor = vec4(vec3(depth), 1.0);
}
