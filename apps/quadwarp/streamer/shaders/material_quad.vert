layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
layout(location = 2) in vec3 aNormal;
layout(location = 3) in vec2 aTexCoords;
layout(location = 4) in vec3 aTexCoords3D;
layout(location = 5) in vec3 aBitangent;

out VertexData {
    vec2 TexCoords;
    vec3 TexCoords3D;
    vec3 FragPos;
    vec3 Color;
    vec3 Normal;
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

void main() {
    vsOut.FragPos = vec3(model * vec4(aPos, 1.0));
    vsOut.Color = aColor;
    vsOut.Normal = normalize(normalMatrix * aNormal);

#ifndef ANDROID
    gl_Position = projection * view * vec4(vsOut.FragPos, 1.0);
#else
    gl_Position = projection[gl_ViewID_OVR] * view[gl_ViewID_OVR] * vec4(vsOut.FragPos, 1.0);
#endif

    vsOut.TexCoords = aTexCoords; // unused
    vsOut.TexCoords3D = aTexCoords3D;
}
