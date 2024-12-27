out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D screenColor;
uniform sampler2D screenDepth;
uniform sampler2D screenPositions;
uniform sampler2D screenNormals;
uniform usampler2D idBuffer;

uniform float near = 0.1;
uniform float far = 1000.0;

float LinearizeDepth(float depth) {
    float z = depth * 2.0 - 1.0; // back to NDC
    return (2.0 * near * far) / (far + near - z * (far - near));
}

void main() {
    float depth = LinearizeDepth(texture(screenDepth, TexCoords).r) / far;
    FragColor = vec4(vec3(depth), 1.0);
}
