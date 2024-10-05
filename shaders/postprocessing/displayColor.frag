out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D screenColor;
uniform sampler2D screenDepth;
uniform sampler2D screenPositions;
uniform sampler2D screenNormals;
uniform sampler2D idBuffer;

uniform bool doToneMapping = true;
uniform float exposure = 1.0;

void main() {
    vec3 color = texture(screenColor, TexCoords).rgb;
    if (doToneMapping) {
        vec3 toneMappedResult = vec3(1.0) - exp(-color.rgb * exposure);
        color = toneMappedResult;
    }
    FragColor = vec4(color, 1.0);
}
