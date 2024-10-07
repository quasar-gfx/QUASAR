layout(location = 0) in uint aID;
layout(location = 1) in vec3 aPos;
layout(location = 2) in vec3 aColor;
layout(location = 3) in vec3 aNormal;
layout(location = 4) in vec2 aTexCoords;
layout(location = 5) in vec3 aTangent;
layout(location = 6) in vec3 aBitangent;

out VertexData {
    flat uint VertexID;
    vec3 TexCoords;
    vec3 FragPos;
    vec3 Color;
    vec3 Normal;
    vec3 Tangent;
    vec3 BiTangent;
    vec4 FragPosLightSpace;
} vsOut;

#ifndef ANDROID
uniform mat4 projection;
uniform mat4 view;
#else
layout(num_views = 2) in;

uniform mat4 projection[2];
uniform mat4 view[2];
#endif

uniform mat4 model;
uniform mat3 normalMatrix;
uniform mat4 lightSpaceMatrix;

uniform float pointSize;

void main() {
    vsOut.VertexID = aID;
    vsOut.FragPos = vec3(model * vec4(aPos, 1.0));
    vsOut.Color = aColor;
    vsOut.Normal = normalize(normalMatrix * aNormal);
    vsOut.Tangent = normalize(normalMatrix * aTangent);
    vsOut.BiTangent = normalize(normalMatrix * aBitangent);

    vsOut.FragPosLightSpace = lightSpaceMatrix * vec4(vsOut.FragPos, 1.0);

    gl_PointSize = pointSize;

#ifndef ANDROID
    gl_Position = projection * view * vec4(vsOut.FragPos, 1.0);
#else
    gl_Position = projection[gl_ViewID_OVR] * view[gl_ViewID_OVR] * vec4(vsOut.FragPos, 1.0);
#endif

    float reciprocalW = 1.0 / gl_Position.w;
    vsOut.TexCoords = vec3(aTexCoords * reciprocalW, reciprocalW);
}
