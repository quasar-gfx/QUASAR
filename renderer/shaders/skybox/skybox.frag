layout(location = 0) out vec4 FragColor;
layout(location = 1) out float FragAlpha;
layout(location = 2) out vec4 FragNormal;
layout(location = 3) out uvec4 FragIDs;

in VertexData {
    flat uint DrawID;
    vec3 WorldPos;
} fsIn;

uniform samplerCube environmentMap;

void main() {
    vec3 envColor = textureLod(environmentMap, fsIn.WorldPos, 0.0).rgb;

    FragColor = vec4(envColor, 1.0);
    FragNormal = vec4(0.0/0.0, 0.0/0.0, 0.0/0.0, 1.0); // make NaN
    FragIDs = uvec4(fsIn.DrawID, gl_PrimitiveID, 0, 1);
}
