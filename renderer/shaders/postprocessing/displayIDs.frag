out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D screenColor;
uniform sampler2D screenDepth;
uniform sampler2D screenNormals;
uniform usampler2D idBuffer;

uniform bool showObjectIDs = true;

void main() {
    uvec3 ids = texture(idBuffer, TexCoords).rgb;
    unsigned int id = showObjectIDs ? ids.r : ids.g;

    vec3 col = vec3( (id % 256) / 255.0,
                    ((id / 256) % 256) / 255.0,
                    ((id / (256 * 256)) % 256) / 255.0 );
    FragColor = vec4(col, 1.0);
}
