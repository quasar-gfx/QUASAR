// Lights
struct AmbientLight {
    vec3 color;
    float intensity;
};

struct DirectionalLight {
    vec3 direction;

    vec3 color;
    float intensity;
};

struct PointLight {
    vec3 position;
    int shadowIndex;

    vec3 color;
    float intensity;

    float constant;
    float linear;
    float quadratic;
    float farPlane;
};
