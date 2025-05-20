// Lights
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
    float farPlane;
};

struct PBRInfo {
    vec3 N;
    vec3 V;
    vec3 R;
    vec3 albedo;
    float metallic;
    float roughness;
    vec3 F0;
};
