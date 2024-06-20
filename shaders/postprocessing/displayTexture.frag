out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D screenPositions;
uniform sampler2D screenNormals;
uniform sampler2D idBuffer;
uniform sampler2D screenColor;
uniform sampler2D screenDepth;

uniform sampler2D tex;

void main() {
    FragColor = vec4(texture(tex, TexCoords).rgb, 1.0);
}
