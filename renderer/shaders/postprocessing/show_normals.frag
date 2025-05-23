out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D screenColor;
uniform sampler2D screenDepth;
uniform sampler2D screenNormals;
uniform sampler2D screenPositions;
uniform usampler2D idBuffer;

void main() {
    vec3 normals = texture(screenNormals, TexCoords).xyz;
    FragColor = vec4(normals, 1.0);
}
