layout(location = 0) out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D screenColor;
uniform sampler2D screenDepth;
uniform sampler2D screenNormals;
uniform sampler2D screenPositions;
uniform usampler2D idTexture;

uniform sampler2D tex;

void main() {
    FragColor = vec4(texture(tex, TexCoord).rgb, 1.0);
}
