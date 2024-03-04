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
    float shininess;
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

#define NR_POINT_LIGHTS 4

uniform AmbientLight ambientLight;
uniform DirectionalLight directionalLight;
uniform PointLight pointLights[NR_POINT_LIGHTS];

uniform samplerCube skybox;

vec3 addSkyBoxLight(vec3 normal, vec3 viewDir) {
    vec3 reflectDir = reflect(-viewDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
    vec3 diffuse = texture(material.diffuse, fs_in.TexCoords).rgb;
    vec3 specular = spec * texture(material.specular, fs_in.TexCoords).rrr;
    vec3 color = texture(skybox, reflectDir).rgb;
    return 0.1 * color * (diffuse + specular);
}

vec3 addDirectionalLight(DirectionalLight light, vec3 normal, vec3 viewDir) {
    vec3 lightDir = normalize(-light.direction);
    // diffuse shading
    float diff = max(dot(normal, lightDir), 0.0);
    // specular shading
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
    vec3 diffuse = diff * texture(material.diffuse, fs_in.TexCoords).rgb;
    vec3 specular = spec * texture(material.specular, fs_in.TexCoords).rrr;
    return light.intensity * light.color * (diffuse + specular);
}

vec3 addPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir) {
    vec3 lightDir = normalize(light.position - fragPos);
    // diffuse shading
    float diff = max(dot(normal, lightDir), 0.0);
    // specular shading
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
    // attenuation
    float distance = length(light.position - fragPos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance) + 0.00001);
    // combine results
    vec3 diffuse = diff * texture(material.diffuse, fs_in.TexCoords).rgb;
    vec3 specular = spec * texture(material.specular, fs_in.TexCoords).rrr;
    return attenuation * (light.intensity * light.color * (diffuse + specular));
}

void main() {
    vec3 norm = normalize(fs_in.Normal);
    vec3 viewDir = normalize(viewPos - fs_in.FragPos);

    vec4 col = texture(material.diffuse, fs_in.TexCoords);
    if (col.a < 0.5)
        discard;

    vec3 result = ambientLight.intensity * ambientLight.color * col.rgb;
    result += addDirectionalLight(directionalLight, norm, viewDir);
    for (int i = 0; i < NR_POINT_LIGHTS; i++) {
        result += addPointLight(pointLights[i], norm, fs_in.FragPos, viewDir);
    }
    // result += addSkyBoxLight(norm, viewDir);

    FragColor = vec4(result, 1.0);
}
