layout(location = 0) in uint aID;
layout(location = 1) in vec3 aPos;
layout(location = 2) in vec3 aColor;
layout(location = 3) in vec3 aNormal;
layout(location = 4) in vec2 aTexCoords;
layout(location = 5) in vec3 aTangent;
layout(location = 6) in vec3 aBitangent;

out VertexData {
    flat uint VertexID;
    vec2 TexCoords;
    vec3 FragPos;
    vec3 Color;
    vec3 Normal;
    vec3 Tangent;
    vec3 BiTangent;
    vec4 FragPosLightSpace;
} vsOut;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
uniform mat3 normalMatrix;
uniform mat4 lightSpaceMatrix;

void main() {
    vsOut.VertexID = aID;

    vsOut.TexCoords = aTexCoords;
    vsOut.Color = aColor;
    vsOut.FragPos = vec3(model * vec4(aPos, 1.0));
    vsOut.Normal = normalize(normalMatrix * aNormal);
    vsOut.Tangent = normalize(vec3(model * vec4(aTangent, 1.0)));
    vsOut.BiTangent = normalize(vec3(model * vec4(aBitangent, 1.0)));

    vsOut.FragPosLightSpace = lightSpaceMatrix * vec4(vsOut.FragPos, 1.0);

    gl_PointSize = 5.0;
    gl_Position = projection * view * vec4(vsOut.FragPos, 1.0);
}
