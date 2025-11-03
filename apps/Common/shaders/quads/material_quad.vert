#include "../camera.glsl"

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aTexCoord3D;

#ifdef ANDROID
layout(num_views = 2) in;
#endif

out VertexData {
    flat uint DrawID;
    vec3 TexCoord3D;
    vec3 FragPos;
} vsOut;

uniform uint drawID;

uniform mat4 model;
uniform mat3 normalMatrix;

void main() {
    vsOut.DrawID = drawID;
    vsOut.FragPos = vec3(model * vec4(aPos, 1.0));
    vsOut.TexCoord3D = aTexCoord3D;

#ifndef ANDROID
    gl_Position = camera.projection * camera.view * vec4(vsOut.FragPos, 1.0);
#else
    gl_Position = camera.projection[gl_ViewID_OVR] * camera.view[gl_ViewID_OVR] * vec4(vsOut.FragPos, 1.0);
#endif
}
