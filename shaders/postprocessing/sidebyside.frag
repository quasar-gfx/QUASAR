out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D screenColor;
uniform sampler2D screenDepth;
uniform sampler2D screenPositions;
uniform sampler2D screenNormals;
uniform sampler2D idBuffer;

uniform sampler2D camera1Buffer;
uniform sampler2D camera2Buffer;

void main() {
    vec4 color;
    if (TexCoords.x < 0.5) {
        vec2 adjustedCoords = vec2(TexCoords.x * 2.0, TexCoords.y);
        color = texture(camera1Buffer, adjustedCoords);
    }
    else {
        vec2 adjustedCoords = vec2((TexCoords.x - 0.5) * 2.0, TexCoords.y);
        color = texture(camera2Buffer, adjustedCoords);
    }

    FragColor = color;
}
