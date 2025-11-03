#include <Utils/FileIO.h>
#include <SceneLoader.h>
#include <Primitives/Primitives.h>

using nlohmann::json;
using namespace quasar;

SceneLoader::~SceneLoader() {
    for (auto* material : materials) {
        delete material;
    }
    for (auto* mesh : meshes) {
        delete mesh;
    }
    for (auto* model : models) {
        delete model;
    }
}

static glm::vec3 jvec3(const json& a) {
    glm::vec3 v{0.f, 0.f, 0.f};
    if (a.is_array() && a.size() >= 3) {
        v[0] = a[0].get<float>();
        v[1] = a[1].get<float>();
        v[2] = a[2].get<float>();
    }

    return v;
}

void SceneLoader::loadScene(const Path& filename, Scene& scene, PerspectiveCamera& camera) {
    uint size;
    std::string sceneJSON = FileIO::loadFromTextFile(filename, &size);
    if (size == 0) {
        throw std::runtime_error("Scene file is empty: " + filename.str());
    }

    auto j = json::parse(sceneJSON);

    spdlog::info("Loading scene: {}", filename.str());
    parse(j, scene, camera);
}

void SceneLoader::clear(Scene& scene) {
    scene.clear();
    meshes.clear();
    materials.clear();
}

void SceneLoader::parseSkybox(const json& j, Scene& scene) {
    TextureFileCreateParams hdrTextureParams{
        .type = GL_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR,
    };

    CubeMapCreateParams skyBoxParams{};

    if (j.contains("path")) {
        hdrTextureParams.path = j.at("path").get<std::string>();
    }

    if (j.contains("flipTextureY")) {
        hdrTextureParams.flipTextureY = j.at("flipTextureY").get<bool>();
    }

    if (j.contains("width")) {
        skyBoxParams.width = static_cast<uint>(j.at("width").get<float>());
    }

    if (j.contains("height")) {
        skyBoxParams.height = static_cast<uint>(j.at("height").get<float>());
    }

    if (j.contains("HDR")) {
        skyBoxParams.type = j.at("HDR").get<bool>() ? CubeMapType::HDR : CubeMapType::STANDARD;
    }

    Texture hdrTexture = Texture(hdrTextureParams);

    SkyBox* skybox = new SkyBox(skyBoxParams, hdrTexture);

    scene.setEnvMap(skybox);
}

void SceneLoader::parseMaterial(const json& j, Scene& scene) {
    LitMaterialCreateParams params{};

    if (j.contains("albedoTexturePath")) {
        params.albedoTexturePath = j.at("albedoTexturePath").get<std::string>();
    }

    if (j.contains("normalTexturePath")) {
        params.normalTexturePath = j.at("normalTexturePath").get<std::string>();
    }

    if (j.contains("metallicTexturePath")) {
        params.metallicTexturePath = j.at("metallicTexturePath").get<std::string>();
    }

    if (j.contains("roughnessTexturePath")) {
        params.roughnessTexturePath = j.at("roughnessTexturePath").get<std::string>();
    }

    if (j.contains("aoTexturePath")) {
        params.aoTexturePath = j.at("aoTexturePath").get<std::string>();
    }

    if (j.contains("alphaMode")) {
        auto alphaMode = j.at("alphaMode").get<std::string>();
        if (alphaMode == "opaque") {
            params.alphaMode = Material::AlphaMode::OPAQUE;
        }
        else if (alphaMode == "masked") {
            params.alphaMode = Material::AlphaMode::MASKED;
        }
        else if (alphaMode == "transparent") {
            params.alphaMode = Material::AlphaMode::TRANSPARENT;
        }
    }

    if (j.contains("maskThreshold")) {
        params.maskThreshold = j.at("maskThreshold").get<float>();
    }

    auto* material = new LitMaterial(params);
    materials.push_back(material);
}

void SceneLoader::parseMaterials(const json& j, Scene& scene) {
    for (auto& m : j) {
        parseMaterial(m, scene);
    }
}

void SceneLoader::parseModel(const json& j, Scene& scene) {
    std::string name = "Model" + std::to_string(models.size());

    ModelCreateParams params{};

    if (j.contains("name")) {
        name = j.at("name").get<std::string>();
    }

    if (j.contains("path")) {
        params.path = j.at("path").get<std::string>();
    }

    if (j.contains("IBL")) {
        params.IBL = j.at("IBL").get<float>();
    }

    if (j.contains("flipTextureY")) {
        params.flipTextureY = j.at("flipTextureY").get<bool>();
    }

    if (j.contains("gammaCorrected")) {
        params.gammaCorrected = j.at("gammaCorrected").get<bool>();
    }

    if (j.contains("material")) {
        int materialIdx = static_cast<int>(j.at("material").get<float>());
        if (materialIdx < 0 || materialIdx >= static_cast<int>(materials.size())) {
            throw std::runtime_error("Material index out of bounds for Mesh 0");
        }

        params.material = materials[materialIdx];
    }

    auto* model = new Model(params);
    model->setName(name);

    // Wrapper node
    auto* node = new Node();
    node->addChildNode(model);

    if (j.contains("position")) {
        node->setPosition(jvec3(j.at("position")));
    }

    if (j.contains("rotation")) {
        node->setRotationEuler(jvec3(j.at("rotation")));
    }

    if (j.contains("scale")) {
        node->setScale(jvec3(j.at("scale")));
    }

    if (j.contains("wireframe")) {
        node->wireframe = j.at("wireframe").get<bool>();
    }

    if (j.contains("pointcloud")) {
        bool pointcloud = j.at("pointcloud").get<bool>();
        node->primitiveType = pointcloud ? GL_POINTS : GL_TRIANGLES;
    }

    models.push_back(model);
    scene.addChildNode(node);
}

void SceneLoader::parseModels(const json& j, Scene& scene) {
    for (auto& m : j) {
        parseModel(m, scene);
    }
}

void SceneLoader::parseMesh(const json& j, Scene& scene) {
    std::string name = "Mesh" + std::to_string(meshes.size());

    MeshDataCreateParams params{};

    if (j.contains("name")) {
        name = j.at("name").get<std::string>();
    }

    if (j.contains("material")) {
        int materialIdx = static_cast<int>(j.at("material").get<float>());
        if (materialIdx < 0 || materialIdx >= static_cast<int>(materials.size())) {
            throw std::runtime_error("Material index out of bounds for Mesh 0");
        }

        params.material = materials[materialIdx];
    }

    if (j.contains("IBL")) {
        params.IBL = j.at("IBL").get<float>();
    }

    std::string meshType = "cube";
    if (j.contains("type")) {
        meshType = j.at("type").get<std::string>();
    }

    Mesh* mesh = nullptr;
    if (meshType == "cube") {
        mesh = new Cube(params);
    }
    else if (meshType == "sphere") {
        mesh = new Sphere(params);
    }
    else if (meshType == "plane") {
        mesh = new Plane(params);
    }

    meshes.push_back(mesh);
}

void SceneLoader::parseMeshes(const json& j, Scene& scene) {
    for (auto& m : j) {
        parseMesh(m, scene);
    }
}

void SceneLoader::parseNode(const json& j, Scene& scene) {
    Node* node = new Node();

    if (j.contains("mesh")) {
        int idx = static_cast<int>(j.at("mesh").get<float>());
        if (idx < 0 || idx >= static_cast<int>(meshes.size())) {
            delete node;
            throw std::runtime_error("Mesh index out of bounds for Node 0");
        }

        node->addEntity(meshes[idx]);
    }

    if (j.contains("position")) {
        node->setPosition(jvec3(j.at("position")));
    }

    if (j.contains("rotation")) {
        node->setRotationEuler(jvec3(j.at("rotation")));
    }

    if (j.contains("scale")) {
        node->setScale(jvec3(j.at("scale")));
    }

    if (j.contains("wireframe")) {
        node->wireframe = j.at("wireframe").get<bool>();
    }

    if (j.contains("pointcloud")) {
        bool pointcloud = j.at("pointcloud").get<bool>();
        node->primitiveType = pointcloud ? GL_POINTS : GL_TRIANGLES;
    }

    scene.addChildNode(node);
}

void SceneLoader::parseNodes(const json& j, Scene& scene) {
    for (auto& n : j) {
        parseNode(n, scene);
    }
}

void SceneLoader::parseCamera(const json& j, Scene& scene, PerspectiveCamera& camera) {
    if (j.contains("fov")) {
        camera.setFovyDegrees(j.at("fov").get<float>());
    }

    if (j.contains("near")) {
        camera.setNear(j.at("near").get<float>());
    }

    if (j.contains("far")) {
        camera.setFar(j.at("far").get<float>());
    }

    if (j.contains("position")) {
        camera.setPosition(jvec3(j.at("position")));
    }

    if (j.contains("rotation")) {
        camera.setRotationEuler(jvec3(j.at("rotation")));
    }

    camera.updateProjectionMatrix();
    camera.updateViewMatrix();
}

void SceneLoader::parseAmbientLight(const json& j, Scene& scene) {
    AmbientLightCreateParams params{};

    if (j.contains("color")) {
        params.color = jvec3(j.at("color"));
    }

    if (j.contains("intensity")) {
        params.intensity = j.at("intensity").get<float>();
    }

    auto ambientLight = new AmbientLight(params);

    scene.setAmbientLight(ambientLight);
}

void SceneLoader::parseDirectionalLight(const json& j, Scene& scene) {
    DirectionalLightCreateParams params{};

    if (j.contains("color")) {
        params.color = jvec3(j.at("color"));
    }

    if (j.contains("direction")) {
        params.direction = jvec3(j.at("direction"));
    }

    if (j.contains("distance")) {
        params.distance = j.at("distance").get<float>();
    }

    if (j.contains("intensity")) {
        params.intensity = j.at("intensity").get<float>();
    }

    if (j.contains("orthoBoxSize")) {
        params.orthoBoxSize = j.at("orthoBoxSize").get<float>();
    }

    if (j.contains("shadowFar")) {
        params.shadowFar = j.at("shadowFar").get<float>();
    }

    if (j.contains("shadowNear")) {
        params.shadowNear = j.at("shadowNear").get<float>();
    }

    if (j.contains("shadowMapRes")) {
        params.shadowMapRes = static_cast<uint>(j.at("shadowMapRes").get<float>());
    }

    auto directionalLight = new DirectionalLight(params);

    scene.setDirectionalLight(directionalLight);
}

void SceneLoader::parsePointLight(const json& j, Scene& scene) {
    PointLightCreateParams params{};

    if (j.contains("color")) {
        params.color = jvec3(j.at("color"));
    }

    if (j.contains("position")) {
        params.position = jvec3(j.at("position"));
    }

    if (j.contains("intensity")) {
        params.intensity = j.at("intensity").get<float>();
    }

    if (j.contains("constant")) {
        params.constant = j.at("constant").get<float>();
    }

    if (j.contains("linear")) {
        params.linear = j.at("linear").get<float>();
    }

    if (j.contains("quadratic")) {
        params.quadratic = j.at("quadratic").get<float>();
    }

    if (j.contains("shadowFar")) {
        params.shadowFar = j.at("shadowFar").get<float>();
    }

    if (j.contains("shadowNear")) {
        params.shadowNear = j.at("shadowNear").get<float>();
    }

    if (j.contains("shadowFov")) {
        params.shadowFov = j.at("shadowFov").get<float>();
    }

    if (j.contains("shadowMapRes")) {
        params.shadowMapRes = static_cast<uint>(j.at("shadowMapRes").get<float>());
    }

    if (j.contains("debug")) {
        params.debug = j.at("debug").get<bool>();
    }

    auto pointLight = new PointLight(params);

    scene.addPointLight(pointLight);
}

void SceneLoader::parsePointLights(const json& j, Scene& scene) {
    for (auto& p : j) {
        parsePointLight(p, scene);
    }
}

void SceneLoader::parseAnimation(const json& j, Scene& scene) {
    std::string nodeName;
    std::string property;

    glm::vec3 fromPosition{0.0f, 0.0f, 0.0f}, toPosition{0.0f, 0.0f, 0.0f};
    glm::vec3 fromRotation{0.0f, 0.0f, 0.0f}, toRotation{0.0f, 0.0f, 0.0f};
    glm::vec3 fromScale{1.0f, 1.0f, 1.0f}, toScale{1.0f, 1.0f, 1.0f};

    float duration = 1.0f;
    float delay = 0.0f;

    bool reverse = false;
    bool loop = false;

    if (j.contains("node")) {
        nodeName = j.at("node").get<std::string>();
    }

    if (j.contains("property")) {
        property = j.at("property").get<std::string>();
    }

    if (j.contains("fromPosition")) {
        fromPosition = jvec3(j.at("fromPosition"));
    }

    if (j.contains("toPosition")) {
        toPosition = jvec3(j.at("toPosition"));
    }

    if (j.contains("fromRotation")) {
        fromRotation = jvec3(j.at("fromRotation"));
    }

    if (j.contains("toRotation")) {
        toRotation = jvec3(j.at("toRotation"));
    }

    if (j.contains("fromScale")) {
        fromScale = jvec3(j.at("fromScale"));
    }

    if (j.contains("toScale")) {
        toScale = jvec3(j.at("toScale"));
    }

    if (j.contains("delay")) {
        delay = j.at("delay").get<float>();
    }

    if (j.contains("duration")) {
        duration = j.at("duration").get<float>();
    }

    if (j.contains("reverse")) {
        reverse = j.at("reverse").get<bool>();
    }

    if (j.contains("loop")) {
        loop = j.at("loop").get<bool>();
    }

    Node* node = scene.findNodeByName(nodeName);
    if (node != nullptr) {
        std::shared_ptr<Animation> anim = node->addAnimation();

        anim->addPositionKey(fromPosition, delay);
        anim->addPositionKey(toPosition, delay + duration);
        anim->setPositionProperties(reverse, loop);

        anim->addRotationKey(fromRotation, delay);
        anim->addRotationKey(toRotation, delay + duration);
        anim->setRotationProperties(reverse, loop);

        anim->addScaleKey(fromScale, delay);
        anim->addScaleKey(toScale, delay + duration);
        anim->setScaleProperties(reverse, loop);
    }
    else {
        spdlog::warn("Node not found: {}", nodeName);
    }
}

void SceneLoader::parseAnimations(const json& j, Scene& scene) {
    for (auto& a : j) {
        parseAnimation(a, scene);
    }
}

void SceneLoader::parse(const json& j, Scene& scene, PerspectiveCamera& camera) {
    if (j.contains("skybox")) {
        parseSkybox(j.at("skybox"), scene);
    }

    if (j.contains("materials")) {
        parseMaterials(j.at("materials"), scene);
    }

    if (j.contains("models")) {
        parseModels(j.at("models"), scene);
    }

    if (j.contains("meshes")) {
        parseMeshes(j.at("meshes"), scene);
    }

    if (j.contains("nodes")) {
        parseNodes(j.at("nodes"), scene);
    }

    if (j.contains("camera")) {
        parseCamera(j.at("camera"), scene, camera);
    }

    if (j.contains("ambientLight")) {
        parseAmbientLight(j.at("ambientLight"), scene);
    }

    if (j.contains("directionalLight")) {
        parseDirectionalLight(j.at("directionalLight"), scene);
    }

    if (j.contains("pointLights")) {
        parsePointLights(j.at("pointLights"), scene);
    }

    if (j.contains("animations")) {
        parseAnimations(j.at("animations"), scene);
    }
}
