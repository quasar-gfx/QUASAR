layout(location = 0) in uint aID;
layout(location = 1) in vec3 aPos;

uniform mat4 model;

void main() {
    gl_Position = model * vec4(aPos, 1.0);
}
