#version 330 core
out vec4 FragColor;

in VS_OUT {
    vec3 FragPos;
    vec3 Normal;
    vec2 TexCoords;
} fs_in;

struct Material {
    sampler2D diffuse;
    sampler2D specular;
    sampler2D normal;
    sampler2D height;
};

struct AmbientLight {
    vec3 color;
    float intensity;
};

struct DirectionalLight {
    vec3 color;
    vec3 direction;
    float intensity;
};

struct PointLight {
    vec3 color;
    vec3 position;
    float intensity;
    float constant;
    float linear;
    float quadratic;
};

uniform Material material;

uniform vec3 viewPos;
uniform AmbientLight ambientLight;
uniform DirectionalLight directionalLight;
uniform PointLight pointLight;

vec3 addDirectionalLight(DirectionalLight light, vec3 normal, vec3 viewDir) {
    vec3 lightDir = normalize(-light.direction);
    // diffuse shading
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * vec3(texture(material.diffuse, fs_in.TexCoords));
    return light.intensity * light.color * diffuse;
}

vec3 addPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir) {
    vec3 lightDir = normalize(light.position - fragPos);
    // diffuse shading
    float diff = max(dot(normal, lightDir), 0.0);
    // attenuation
    float distance = length(light.position - fragPos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
    // combine results
    vec3 diffuse = diff * vec3(texture(material.diffuse, fs_in.TexCoords));
    diffuse *= attenuation;
    return light.intensity * light.color * diffuse;
}

void main() {
    vec3 norm = normalize(fs_in.Normal);
    vec3 viewDir = normalize(viewPos - fs_in.FragPos);

    vec4 col = texture(material.diffuse, fs_in.TexCoords);
    if (col.a < 0.5)
        discard;

    vec3 direct = ambientLight.intensity * ambientLight.color * col.rgb;
    vec3 indirect = vec3(0.0);
    indirect += addDirectionalLight(directionalLight, norm, viewDir);
    indirect += addPointLight(pointLight, norm, fs_in.FragPos, viewDir);

    vec3 result = direct + indirect;

    FragColor = vec4(result, 1.0);
}
