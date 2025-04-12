#include "camera.glsl"

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
layout(location = 2) in vec3 aNormal;
layout(location = 3) in vec2 aTexCoords;
layout(location = 4) in vec3 aTangent;
layout(location = 5) in vec3 aBitangent;

#ifdef ANDROID
layout(num_views = 2) in;
#endif

out VertexData {
    flat unsigned int drawID;
    vec2 TexCoords;
    vec3 FragPosView;
    vec3 FragPosWorld;
    vec3 Color;
    vec3 Normal;
    vec3 Tangent;
    vec3 BiTangent;
    vec4 FragPosLightSpace;
} vsOut;

uniform unsigned int drawID;

uniform mat4 model;
uniform mat3 normalMatrix;
uniform mat4 lightSpaceMatrix;

void main() {
    mat4 modelMatrix = model;
#ifndef ANDROID
    mat4 viewMatrix = camera.view;
    mat4 projectionMatrix = camera.projection;
#else
    mat4 viewMatrix = camera.view[gl_ViewID_OVR];
    mat4 projectionMatrix = camera.projection[gl_ViewID_OVR];
#endif

    vec4 worldPos = modelMatrix * vec4(aPos, 1.0);
    vec4 viewPos = viewMatrix * worldPos;

    vsOut.drawID = drawID;
    vsOut.TexCoords = aTexCoords;
    vsOut.FragPosView = viewPos.xyz;
    vsOut.FragPosWorld = worldPos.xyz;
    vsOut.Color = aColor;
    vsOut.Normal = normalize(normalMatrix * aNormal);
    vsOut.Tangent = normalize(normalMatrix * aTangent);
    vsOut.BiTangent = normalize(normalMatrix * aBitangent);

    vsOut.FragPosLightSpace = lightSpaceMatrix * vec4(vsOut.FragPosWorld, 1.0);

    gl_Position = projectionMatrix * viewMatrix * vec4(vsOut.FragPosWorld, 1.0);
}
