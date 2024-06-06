layout(location = 0) in uint aID;
layout(location = 1) in vec3 aPos;

uniform mat4 lightSpaceMatrix;

void main() {
    gl_Position = lightSpaceMatrix * vec4(aPos, 1.0);
}
