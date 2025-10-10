layout(location = 0) out vec4 FragColor;
layout(location = 1) out float FragAlpha;
layout(location = 2) out vec3 FragNormal;
layout(location = 3) out uvec4 FragIDs;

in vec2 TexCoord;

uniform sampler2D peelingLayersColor[MAX_LAYERS];
uniform sampler2D peelingLayersAlpha[MAX_LAYERS];

void main() {
    vec4 color = vec4(0.0);
    for (int i = 0; i < MAX_LAYERS; i++) {
        vec3 layerColor = texture(peelingLayersColor[i], TexCoord).rgb;
        float layerAlpha = texture(peelingLayersAlpha[i], TexCoord).r;

        color.rgb += layerColor * (1.0 - color.a);
        color.a   += layerAlpha * (1.0 - color.a);
    }

    FragColor = color;
    FragAlpha = color.a;
}
