layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTexCoords;

#ifdef ANDROID
layout(num_views = 2) in;
#endif

out vec2 TexCoords;

void main() {
    TexCoords = aTexCoords;
    gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
}
