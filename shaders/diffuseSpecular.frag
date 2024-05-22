#version 410 core
layout(location = 0) out vec4 positionBuffer;
layout(location = 1) out vec4 normalsBuffer;
layout(location = 2) out vec4 idBuffer;
layout(location = 3) out vec4 FragColor;

in VertexData {
    flat uint VertexID;
    vec2 TexCoords;
    vec3 FragPos;
    vec3 Normal;
    vec3 Tangent;
    vec3 BiTangent;
    vec4 FragPosLightSpace;
} fsIn;

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

#define NUM_POINT_LIGHTS 4

uniform AmbientLight ambientLight;
uniform DirectionalLight directionalLight;
uniform PointLight pointLights[NUM_POINT_LIGHTS];

// material textures
uniform sampler2D diffuseMap; // 0
uniform sampler2D specularMap; // 1

uniform samplerCube environmentMap;

uniform float shininess;

uniform bool transparent;

uniform vec3 camPos;

vec3 addSkyBoxLight(vec3 normal, vec3 viewDir) {
    vec3 reflectDir = reflect(-viewDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 diffuse = texture(diffuseMap, fsIn.TexCoords).rgb;
    vec3 specular = spec * texture(specularMap, fsIn.TexCoords).rrr;
    vec3 color = texture(environmentMap, reflectDir).rgb;
    return 0.1 * color * (diffuse + specular);
}

vec3 addDirectionalLight(DirectionalLight light, vec3 normal, vec3 viewDir) {
    if (light.intensity <= 0.0)
        return vec3(0.0);

    vec3 lightDir = normalize(-light.direction);
    // diffuse shading
    float diff = max(dot(normal, lightDir), 0.0);
    // specular shading
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 diffuse = diff * texture(diffuseMap, fsIn.TexCoords).rgb;
    vec3 specular = spec * texture(specularMap, fsIn.TexCoords).rrr;
    return light.intensity * light.color * (diffuse + specular);
}

vec3 addPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir) {
    if (light.intensity <= 0.0)
        return vec3(0.0);

    vec3 lightDir = normalize(light.position - fragPos);
    // diffuse shading
    float diff = max(dot(normal, lightDir), 0.0);
    // specular shading
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    // attenuation
    float distance = length(light.position - fragPos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
    // combine results
    vec3 diffuse = diff * texture(diffuseMap, fsIn.TexCoords).rgb;
    vec3 specular = spec * texture(specularMap, fsIn.TexCoords).rrr;
    return attenuation * (light.intensity * light.color * (diffuse + specular));
}

void main() {
    vec4 color = texture(diffuseMap, fsIn.TexCoords);
    float alpha = (transparent) ? color.a : 1.0;
    if (alpha < 0.1)
        discard;

    vec3 norm = normalize(fsIn.Normal);
    vec3 viewDir = normalize(camPos - fsIn.FragPos);

    vec3 result = ambientLight.intensity * ambientLight.color * color.rgb;
    result += addDirectionalLight(directionalLight, norm, viewDir);
    for (int i = 0; i < NUM_POINT_LIGHTS; i++) {
        result += addPointLight(pointLights[i], norm, fsIn.FragPos, viewDir);
    }
    // result += addSkyBoxLight(norm, viewDir);

    positionBuffer = vec4(fsIn.FragPos, 1.0);
    normalsBuffer = vec4(normalize(fsIn.Normal), 1.0);
    idBuffer = vec4(fsIn.VertexID, 0.0, 0.0, 0.0);
    FragColor = vec4(result, alpha);
}
