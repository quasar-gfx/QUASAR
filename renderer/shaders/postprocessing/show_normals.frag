layout(location = 0) out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D screenColor;
uniform sampler2D screenDepth;
uniform sampler2D screenNormals;
uniform sampler2D screenPositions;
uniform usampler2D idTexture;

void main() {
    vec3 normals = texture(screenNormals, TexCoord).xyz;
    FragColor = vec4(normals, 1.0);
}
