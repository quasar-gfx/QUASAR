layout(location = 0) out vec4 FragColor;
layout(location = 1) out vec4 FragPosition;
layout(location = 2) out vec4 FragNormal;
layout(location = 3) out vec4 FragIDs;

in vec3 WorldPos;

uniform samplerCube environmentMap;

void main() {
    vec3 envColor = textureLod(environmentMap, WorldPos, 0.0).rgb;

    FragColor = vec4(envColor, 1.0);
    FragPosition = vec4(WorldPos, 1.0);
    FragNormal = vec4(0.0/0.0, 0.0/0.0, 0.0/0.0, 1.0); // make NaN
    FragIDs = vec4(gl_PrimitiveID, 0.0, 0.0, 0.0);
}
