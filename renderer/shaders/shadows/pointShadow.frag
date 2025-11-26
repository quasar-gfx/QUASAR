in vec4 FragPos;

uniform vec3 lightPos;
uniform float shadowFar;

void main() {
    float lightDistance = length(FragPos.xyz - lightPos);

    // Map to [0;1] range by dividing by shadowFar
    lightDistance = lightDistance / shadowFar;

    // Write this as modified depth
    gl_FragDepth = lightDistance;
}
