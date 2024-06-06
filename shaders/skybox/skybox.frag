layout(location = 0) out vec4 positionBuffer;
layout(location = 1) out vec4 normalsBuffer;
layout(location = 2) out vec4 idBuffer;
layout(location = 3) out vec4 FragColor;

in vec3 WorldPos;

uniform samplerCube environmentMap;

void main() {
    vec3 envColor = textureLod(environmentMap, WorldPos, 0.0).rgb;
    FragColor = vec4(envColor, 1.0);
}
