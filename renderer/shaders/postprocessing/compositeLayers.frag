layout(location = 0) out vec4 FragColor;
layout(location = 1) out vec4 FragPosition;
layout(location = 2) out vec4 FragNormal;
layout(location = 3) out uvec4 FragIDs;

in vec2 TexCoords;

uniform sampler2D peelingLayers[MAX_LAYERS];

void main() {
    vec4 color = vec4(0.0);
    for (int i = 0; i < MAX_LAYERS; i++) {
        vec4 layerColor = texture(peelingLayers[i], TexCoords);
        color.rgb += layerColor.rgb * (1.0 - color.a);
        color.a += layerColor.a * (1.0 - color.a);
    }

    FragColor = color;
}
