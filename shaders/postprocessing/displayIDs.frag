out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D screenPositions;
uniform sampler2D screenNormals;
uniform sampler2D idBuffer;
uniform sampler2D screenColor;
uniform sampler2D screenDepth;

uniform float exposure = 1.0;

void main() {
    uint id = uint(texture(idBuffer, TexCoords).r);
    vec3 col = vec3((id % 256) / 255.0,
                    ((id / 256) % 256) / 255.0,
                    ((id / (256 * 256)) % 256) / 255.0);
    FragColor = vec4(col, 1.0f);
}
