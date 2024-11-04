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
    vec2 TexCoords;
    vec3 FragPos;
    vec3 Color;
    vec3 Normal;
    vec3 Tangent;
    vec3 BiTangent;
    vec4 FragPosLightSpace;
} vsOut;

uniform struct Camera {
#ifndef ANDROID
    mat4 projection;
    mat4 view;
#else
    mat4 projection[2];
    mat4 view[2];
#endif
    vec3 position;
    float near;
    float far;
} camera;

uniform mat4 model;
uniform mat3 normalMatrix;
uniform mat4 lightSpaceMatrix;

void main() {
    vsOut.TexCoords = aTexCoords;
    vsOut.FragPos = vec3(model * vec4(aPos, 1.0));
    vsOut.Color = aColor;
    vsOut.Normal = normalize(normalMatrix * aNormal);
    vsOut.Tangent = normalize(normalMatrix * aTangent);
    vsOut.BiTangent = normalize(normalMatrix * aBitangent);

    vsOut.FragPosLightSpace = lightSpaceMatrix * vec4(vsOut.FragPos, 1.0);

#ifndef ANDROID
    gl_Position = camera.projection * camera.view * vec4(vsOut.FragPos, 1.0);
#else
    gl_Position = camera.projection[gl_ViewID_OVR] * camera.view[gl_ViewID_OVR] * vec4(vsOut.FragPos, 1.0);
#endif
}
