layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aTexCoords3D;

#ifdef ANDROID
layout(num_views = 2) in;
#endif

out VertexData {
    flat uint drawID;
    vec3 TexCoords3D;
    vec3 FragPos;
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
    float fovy;
    float near;
    float far;
} camera;

uniform uint drawID;

uniform mat4 model;
uniform mat3 normalMatrix;

void main() {
    vsOut.drawID = drawID;
    vsOut.FragPos = vec3(model * vec4(aPos, 1.0));
    vsOut.TexCoords3D = aTexCoords3D;

#ifndef ANDROID
    gl_Position = camera.projection * camera.view * vec4(vsOut.FragPos, 1.0);
#else
    gl_Position = camera.projection[gl_ViewID_OVR] * camera.view[gl_ViewID_OVR] * vec4(vsOut.FragPos, 1.0);
#endif
}
