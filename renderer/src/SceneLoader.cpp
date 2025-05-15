#include <spdlog/spdlog.h>

#include <Utils/FileIO.h>
#include <SceneLoader.h>

using namespace quasar;

class JsonParser {
public:
    JsonParser() = default;
    ~JsonParser() {
        delete[] tokens;
    }

    bool readJson(const char* json, size_t size) {
        jsmn_init(&parser);

        int numTokens = jsmn_parse(&parser, json, size, NULL, 0);
        tokens = new jsmntok_t[numTokens];

        jsmn_init(&parser);
        jsmn_parse(&parser, json, size, tokens, numTokens);

        return numTokens >= 0;
    }

    jsmntok_t* getTokens() {
        return tokens;
    }

private:
    jsmn_parser parser;
    jsmntok_t* tokens;
};

Mesh* SceneLoader::findMeshByName(const std::string& name) {
    auto it = meshIndices.find(name);
    if (it != meshIndices.end()) {
        return meshes[it->second];
    }
    return nullptr;
}

Model* SceneLoader::findModelByName(const std::string& name) {
    auto it = modelIndices.find(name);
    if (it != modelIndices.end()) {
        return models[it->second];
    }
    return nullptr;
}

Node* SceneLoader::findNodeByName(const std::string& name) {
    for (auto model : models) {
        Node* node = model->rootNode.findNodeByName(name);
        if (node != nullptr) {
            return node;
        }
    }

    return nullptr;
}

void SceneLoader::loadScene(const std::string& filename, Scene& scene, PerspectiveCamera& camera) {
    uint size;
    std::string sceneJSON = FileIO::loadTextFile(filename, &size);
    if (size == 0) {
        throw std::runtime_error("Scene file is empty: " + filename);
    }

    JsonParser parser;
    if (!parser.readJson(sceneJSON.data(), size)) {
        throw std::runtime_error("Failed to parse scene file " + filename);
    }

    spdlog::info("Loading scene: {}", filename);

    jsmntok_t* tokens = parser.getTokens();
    parse(tokens, 0, sceneJSON.data(), scene, camera);
}

void SceneLoader::clearScene(Scene& scene, PerspectiveCamera& camera) {
    scene.clear();

    for (auto model : models) {
        delete model;
    }
    models.clear();

    for (auto mesh : meshes) {
        delete mesh;
    }
    meshes.clear();

    for (auto material : materials) {
        delete material;
    }
    materials.clear();
}

int SceneLoader::compare(jsmntok_t tok, const char* jsonChunk, const char* str) {
    size_t slen = strlen(str);
    size_t tlen = tok.end - tok.start;
    return (slen == tlen) ? strncmp(jsonChunk + tok.start, str, slen) : 128;
}

int SceneLoader::parse(jsmntok_t const* tokens, int i) {
    int end = i + 1;
    while (i < end) {
        switch (tokens[i].type) {
            case JSMN_OBJECT:
                end += tokens[i].size * 2;
                break;
            case JSMN_ARRAY:
                end += tokens[i].size;
                break;
            case JSMN_PRIMITIVE:
            case JSMN_STRING:
                break;
            default:
                return -1;
        }
        i++;
    }
    return i;
}

int SceneLoader::parseString(jsmntok_t* tokens, int i, const char* json, std::string* str) {
    CHECK_TOKTYPE(tokens[i], JSMN_STRING);
    *str = STR(tokens[i], json);
    return i + 1;
}

int SceneLoader::parseBool(jsmntok_t* tokens, int i, const char* json, bool* b) {
    CHECK_TOKTYPE(tokens[i], JSMN_PRIMITIVE);
    *b = STR(tokens[i], json) == "true";
    return i + 1;
}

int SceneLoader::parseFloat(jsmntok_t* tokens, int i, const char* json, float* num) {
    CHECK_TOKTYPE(tokens[i], JSMN_PRIMITIVE);
    *num = std::stof(STR(tokens[i], json));
    return i + 1;
}

int SceneLoader::parseVec3(jsmntok_t* tokens, int i, const char* json, glm::vec3* vec) {
    CHECK_TOKTYPE(tokens[i], JSMN_ARRAY);

    float num;
    int size = tokens[i++].size;
    for (int j = 0; j < size; j++) {
        i = parseFloat(tokens, i, json, &num);
        (*vec)[j] = num;
    }

    return i;
}

int SceneLoader::parseVec4(jsmntok_t* tokens, int i, const char* json, glm::vec4* vec) {
    CHECK_TOKTYPE(tokens[i], JSMN_ARRAY);

    float num;
    int size = tokens[i++].size;
    for (int j = 0; j < size; j++) {
        i = parseFloat(tokens, i, json, &num);
        (*vec)[j] = num;
    }

    return i;
}

int SceneLoader::parseSkybox(jsmntok_t* tokens, int i, const char* json, Scene& scene, PerspectiveCamera& camera) {
    CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

    TextureFileCreateParams hdrTextureParams{
        .type = GL_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR
    };

    CubeMapCreateParams skyBoxParams{};

    int size = tokens[i++].size;
    for (int j = 0; j < size; j++) {
        const jsmntok_t tok = tokens[i];
        if (compare(tok, json, "path") == 0) {
            i = parseString(tokens, i + 1, json, &hdrTextureParams.path);
        }
        else if (compare(tok, json, "flipVertically") == 0) {
            i = parseBool(tokens, i + 1, json, &hdrTextureParams.flipVertically);
        }
        else if (compare(tok, json, "width") == 0) {
            float width;
            i = parseFloat(tokens, i + 1, json, &width);
            skyBoxParams.width = static_cast<uint>(width);
        }
        else if (compare(tok, json, "height") == 0) {
            float height;
            i = parseFloat(tokens, i + 1, json, &height);
            skyBoxParams.height = static_cast<uint>(height);
        }
        else if (compare(tok, json, "HDR") == 0) {
            bool hdr;
            i = parseBool(tokens, i + 1, json, &hdr);
            skyBoxParams.type = hdr ? CubeMapType::HDR : CubeMapType::STANDARD;
        }
        else {
            i = parse(tokens, i + 1);
        }
        if (i < 0) {
            return i;
        }
    }

    // load the HDR environment map
    Texture hdrTexture = Texture(hdrTextureParams);

    // skybox
    CubeMap* envCubeMap = new CubeMap(skyBoxParams);

    scene.equirectToCubeMap(*envCubeMap, hdrTexture);
    scene.setupIBL(*envCubeMap);
    scene.setEnvMap(envCubeMap);

    return i;
}

int SceneLoader::parseMaterial(jsmntok_t* tokens, int i, const char* json, Scene& scene, PerspectiveCamera& camera) {
    CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

    std::string type;
    LitMaterialCreateParams params{};

    int size = tokens[i++].size;
    for (int j = 0; j < size; j++) {
        const jsmntok_t tok = tokens[i];
        if (compare(tok, json, "type") == 0) {
            i = parseString(tokens, i + 1, json, &type);
        }
        else if (compare(tok, json, "albedoTexturePath") == 0) {
            i = parseString(tokens, i + 1, json, &params.albedoTexturePath);
        }
        else if (compare(tok, json, "normalTexturePath") == 0) {
            i = parseString(tokens, i + 1, json, &params.normalTexturePath);
        }
        else if (compare(tok, json, "metallicTexturePath") == 0) {
            i = parseString(tokens, i + 1, json, &params.metallicTexturePath);
        }
        else if (compare(tok, json, "roughnessTexturePath") == 0) {
            i = parseString(tokens, i + 1, json, &params.roughnessTexturePath);
        }
        else if (compare(tok, json, "aoTexturePath") == 0) {
            i = parseString(tokens, i + 1, json, &params.aoTexturePath);
        }
        else if (compare(tok, json, "alphaMode") == 0) {
            std::string alphaMode;
            i = parseString(tokens, i + 1, json, &alphaMode);
            if (alphaMode == "opaque") {
                params.alphaMode = AlphaMode::OPAQUE;
            }
            else if (alphaMode == "masked") {
                params.alphaMode = AlphaMode::MASKED;
            }
            else if (alphaMode == "transparent") {
                params.alphaMode = AlphaMode::TRANSPARENT;
            }
        }
        else if (compare(tok, json, "maskThreshold") == 0) {
            i = parseFloat(tokens, i + 1, json, &params.maskThreshold);
        }
        else {
            i = parse(tokens, i + 1);
        }
        if (i < 0) {
            return i;
        }
    }

    auto material = new LitMaterial(params);
    materials.push_back(material);

    return i;
}

int SceneLoader::parseMaterials(jsmntok_t* tokens, int i, const char* json, Scene& scene, PerspectiveCamera& camera) {
    CHECK_TOKTYPE(tokens[i], JSMN_ARRAY);

    int size = tokens[i++].size;
    for (int j = 0; j < size; j++) {
        i = parseMaterial(tokens, i, json, scene, camera);
    }

    return i;
}

int SceneLoader::parseModel(jsmntok_t* tokens, int i, const char* json, Scene& scene, PerspectiveCamera& camera) {
    CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

    std::string name = "Model" + std::to_string(models.size());
    ModelCreateParams params{};

    int size = tokens[i++].size;
    for (int j = 0; j < size; j++) {
        const jsmntok_t tok = tokens[i];
        if (compare(tok, json, "name") == 0) {
            i = parseString(tokens, i + 1, json, &name);
        }
        else if (compare(tok, json, "path") == 0) {
            i = parseString(tokens, i + 1, json, &params.path);
        }
        else if (compare(tok, json, "IBL") == 0) {
            i = parseFloat(tokens, i + 1, json, &params.IBL);
        }
        else if (compare(tok, json, "flipTextures") == 0) {
            i = parseBool(tokens, i + 1, json, &params.flipTextures);
        }
        else if (compare(tok, json, "gammaCorrected") == 0) {
            i = parseBool(tokens, i + 1, json, &params.gammaCorrected);
        }
        else if (compare(tok, json, "flipTextures") == 0) {
            i = parseBool(tokens, i + 1, json, &params.flipTextures);
        }
        else if (compare(tok, json, "material") == 0) {
            float materialIdxFloat;
            i = parseFloat(tokens, i + 1, json, &materialIdxFloat);
            int materialIdx = static_cast<int>(materialIdxFloat);
            if (materialIdx < 0 || materialIdx >= materials.size()) {
                throw std::runtime_error("Material index out of bounds for Mesh " + std::to_string(j));
                return -1;
            }
            params.material = materials[materialIdx];
        }
        else {
            i = parse(tokens, i + 1);
        }
        if (i < 0) {
            return i;
        }
    }

    auto model = new Model(params);
    models.push_back(model);
    modelIndices[name] = models.size() - 1;

    return i;
}

int SceneLoader::parseModels(jsmntok_t* tokens, int i, const char* json, Scene& scene, PerspectiveCamera& camera) {
    CHECK_TOKTYPE(tokens[i], JSMN_ARRAY);

    int size = tokens[i++].size;
    for (int j = 0; j < size; j++) {
        i = parseModel(tokens, i, json, scene, camera);
    }

    return i;
}

int SceneLoader::parseMesh(jsmntok_t* tokens, int i, const char* json, Scene& scene, PerspectiveCamera& camera) {
    CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

    std::string meshType = "cube";
    std::string name = "Mesh" + std::to_string(meshes.size());
    MeshDataCreateParams params{};

    int size = tokens[i++].size;
    for (int j = 0; j < size; j++) {
        const jsmntok_t tok = tokens[i];
        if (compare(tok, json, "name") == 0) {
            i = parseString(tokens, i + 1, json, &name);
        }
        else if (compare(tok, json, "type") == 0) {
            i = parseString(tokens, i + 1, json, &meshType);
        }
        else if (compare(tok, json, "material") == 0) {
            float materialIdxFloat;
            i = parseFloat(tokens, i + 1, json, &materialIdxFloat);
            int materialIdx = static_cast<int>(materialIdxFloat);
            if (materialIdx < 0 || materialIdx >= materials.size()) {
                throw std::runtime_error("Material index out of bounds for Mesh " + std::to_string(j));
                return -1;
            }
            params.material = materials[materialIdx];
        }
        else if (compare(tok, json, "IBL") == 0) {
            i = parseFloat(tokens, i + 1, json, &params.IBL);
        }
        else {
            i = parse(tokens, i + 1);
        }
        if (i < 0) {
            return i;
        }
    }

    Mesh* mesh;
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
    meshIndices[name] = meshes.size() - 1;

    return i;
}

int SceneLoader::parseMeshes(jsmntok_t* tokens, int i, const char* json, Scene& scene, PerspectiveCamera& camera) {
    CHECK_TOKTYPE(tokens[i], JSMN_ARRAY);

    int size = tokens[i++].size;
    for (int j = 0; j < size; j++) {
        i = parseMesh(tokens, i, json, scene, camera);
    }

    return i;
}

int SceneLoader::parseNode(jsmntok_t* tokens, int i, const char* json, Scene& scene, PerspectiveCamera& camera) {
    CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

    Node* node = new Node();

    int size = tokens[i++].size;
    for (int j = 0; j < size; j++) {
        const jsmntok_t tok = tokens[i];
        if (compare(tok, json, "model") == 0) {
            float modelIdxFloat;
            i = parseFloat(tokens, i + 1, json, &modelIdxFloat);
            int modelIdx = static_cast<int>(modelIdxFloat);
            if (modelIdx < 0 || modelIdx >= models.size()) {
                throw std::runtime_error("Model index out of bounds for Node " + std::to_string(j));
                return -1;
            }
            node->setEntity(models[modelIdx]);
        }
        else if (compare(tok, json, "mesh") == 0) {
            float meshIdxFloat;
            i = parseFloat(tokens, i + 1, json, &meshIdxFloat);
            int meshIdx = static_cast<int>(meshIdxFloat);
            if (meshIdx < 0 || meshIdx >= meshes.size()) {
                throw std::runtime_error("Mesh index out of bounds for Node " + std::to_string(j));
                return -1;
            }
            node->setEntity(meshes[meshIdx]);
        }
        else if (compare(tok, json, "position") == 0) {
            glm::vec3 position;
            i = parseVec3(tokens, i + 1, json, &position);
            node->setPosition(position);
        }
        else if (compare(tok, json, "rotation") == 0) {
            glm::vec3 rotation;
            i = parseVec3(tokens, i + 1, json, &rotation);
            node->setRotationEuler(rotation);
        }
        else if (compare(tok, json, "scale") == 0) {
            glm::vec3 scale;
            i = parseVec3(tokens, i + 1, json, &scale);
            node->setScale(scale);
        }
        else if (compare(tok, json, "wireframe") == 0) {
            i = parseBool(tokens, i + 1, json, &node->wireframe);
        }
        else if (compare(tok, json, "pointcloud") == 0) {
            bool pointcloud;
            i = parseBool(tokens, i + 1, json, &pointcloud);
            node->primativeType = pointcloud ? GL_POINTS : GL_TRIANGLES;
        }
        else {
            i = parse(tokens, i + 1);
        }
        if (i < 0) {
            return i;
        }
    }

    scene.addChildNode(node);

    return i;
}

int SceneLoader::parseNodes(jsmntok_t* tokens, int i, const char* json, Scene& scene, PerspectiveCamera& camera) {
    CHECK_TOKTYPE(tokens[i], JSMN_ARRAY);

    int size = tokens[i++].size;
    for (int j = 0; j < size; j++) {
        i = parseNode(tokens, i, json, scene, camera);
    }

    return i;
}

int SceneLoader::parseCamera(jsmntok_t* tokens, int i, const char* json, Scene& scene, PerspectiveCamera& camera) {
    CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

    int size = tokens[i++].size;
    for (int j = 0; j < size; j++) {
        const jsmntok_t tok = tokens[i];
        if (compare(tok, json, "fov") == 0) {
            float fov;
            i = parseFloat(tokens, i + 1, json, &fov);
            camera.setFovyDegrees(fov);
        }
        else if (compare(tok, json, "near") == 0) {
            float near;
            i = parseFloat(tokens, i + 1, json, &near);
            camera.setNear(near);
        }
        else if (compare(tok, json, "far") == 0) {
            float far;
            i = parseFloat(tokens, i + 1, json, &far);
            camera.setFar(far);
        }
        else if (compare(tok, json, "position") == 0) {
            glm::vec3 position;
            i = parseVec3(tokens, i + 1, json, &position);
            camera.setPosition(position);
        }
        else if (compare(tok, json, "rotation") == 0) {
            glm::vec3 rotation;
            i = parseVec3(tokens, i + 1, json, &rotation);
            camera.setRotationEuler(rotation);
        }
        else {
            i = parse(tokens, i + 1);
        }
        if (i < 0) {
            return i;
        }
    }

    camera.updateProjectionMatrix();
    camera.updateViewMatrix();

    return i;
}

int SceneLoader::parseAmbientLight(jsmntok_t* tokens, int i, const char* json, Scene& scene, PerspectiveCamera& camera) {
    CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

    AmbientLightCreateParams params{};

    int size = tokens[i++].size;
    for (int j = 0; j < size; j++) {
        const jsmntok_t tok = tokens[i];
        if (compare(tok, json, "color") == 0) {
            i = parseVec3(tokens, i + 1, json, &params.color);
        }
        else if (compare(tok, json, "intensity") == 0) {
            i = parseFloat(tokens, i + 1, json, &params.intensity);
        }
        else {
            i = parse(tokens, i + 1);
        }
        if (i < 0) {
            return i;
        }
    }

    auto ambientLight = new AmbientLight(params);
    scene.setAmbientLight(ambientLight);

    return i;
}

int SceneLoader::parseDirectionalLight(jsmntok_t* tokens, int i, const char* json, Scene& scene, PerspectiveCamera& camera) {
    CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

    DirectionalLightCreateParams params{};

    int size = tokens[i++].size;
    for (int j = 0; j < size; j++) {
        const jsmntok_t tok = tokens[i];
        if (compare(tok, json, "color") == 0) {
            i = parseVec3(tokens, i + 1, json, &params.color);
        }
        else if (compare(tok, json, "direction") == 0) {
            i = parseVec3(tokens, i + 1, json, &params.direction);
        }
        else if (compare(tok, json, "distance") == 0) {
            i = parseFloat(tokens, i + 1, json, &params.distance);
        }
        else if (compare(tok, json, "intensity") == 0) {
            i = parseFloat(tokens, i + 1, json, &params.intensity);
        }
        else if (compare(tok, json, "orthoBoxSize") == 0) {
            i = parseFloat(tokens, i + 1, json, &params.orthoBoxSize);
        }
        else if (compare(tok, json, "shadowFar") == 0) {
            i = parseFloat(tokens, i + 1, json, &params.shadowFar);
        }
        else if (compare(tok, json, "shadowNear") == 0) {
            i = parseFloat(tokens, i + 1, json, &params.shadowNear);
        }
        else if (compare(tok, json, "shadowMapRes") == 0) {
            float shadowMapRes;
            i = parseFloat(tokens, i + 1, json, &shadowMapRes);
            params.shadowMapRes = static_cast<uint>(shadowMapRes);
        }
        else {
            i = parse(tokens, i + 1);
        }
        if (i < 0) {
            return i;
        }
    }

    auto directionalLight = new DirectionalLight(params);
    scene.setDirectionalLight(directionalLight);

    return i;
}

int SceneLoader::parsePointLight(jsmntok_t* tokens, int i, const char* json, Scene& scene, PerspectiveCamera& camera) {
    CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

    PointLightCreateParams params{};

    int size = tokens[i++].size;
    for (int j = 0; j < size; j++) {
        const jsmntok_t tok = tokens[i];
        if (compare(tok, json, "color") == 0) {
            i = parseVec3(tokens, i + 1, json, &params.color);
        }
        else if (compare(tok, json, "position") == 0) {
            i = parseVec3(tokens, i + 1, json, &params.position);
        }
        else if (compare(tok, json, "intensity") == 0) {
            i = parseFloat(tokens, i + 1, json, &params.intensity);
        }
        else if (compare(tok, json, "constant") == 0) {
            i = parseFloat(tokens, i + 1, json, &params.constant);
        }
        else if (compare(tok, json, "linear") == 0) {
            i = parseFloat(tokens, i + 1, json, &params.linear);
        }
        else if (compare(tok, json, "quadratic") == 0) {
            i = parseFloat(tokens, i + 1, json, &params.quadratic);
        }
        else if (compare(tok, json, "shadowFar") == 0) {
            i = parseFloat(tokens, i + 1, json, &params.shadowFar);
        }
        else if (compare(tok, json, "shadowNear") == 0) {
            i = parseFloat(tokens, i + 1, json, &params.shadowNear);
        }
        else if (compare(tok, json, "shadowFov") == 0) {
            i = parseFloat(tokens, i + 1, json, &params.shadowFov);
        }
        else if (compare(tok, json, "shadowMapRes") == 0) {
            float shadowMapRes;
            i = parseFloat(tokens, i + 1, json, &shadowMapRes);
            params.shadowMapRes = static_cast<uint>(shadowMapRes);
        }
        else if (compare(tok, json, "debug") == 0) {
            i = parseBool(tokens, i + 1, json, &params.debug);
        }
        else {
            i = parse(tokens, i + 1);
        }
        if (i < 0) {
            return i;
        }
    }

    auto pointLight = new PointLight(params);
    scene.addPointLight(pointLight);

    return i;
}

int SceneLoader::parsePointLights(jsmntok_t* tokens, int i, const char* json, Scene& scene, PerspectiveCamera& camera) {
    CHECK_TOKTYPE(tokens[i], JSMN_ARRAY);

    int size = tokens[i++].size;
    for (int j = 0; j < size; j++) {
        i = parsePointLight(tokens, i, json, scene, camera);
    }

    return i;
}

int SceneLoader::parseAnimation(jsmntok_t* tokens, int i, const char* json, Scene& scene, PerspectiveCamera& camera) {
    CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

    std::string nodeName;
    std::string property;
    glm::vec3 fromPosition{0.0f, 0.0f, 0.0f}, toPosition{0.0f, 0.0f, 0.0f};
    glm::vec3 fromRotation{0.0f, 0.0f, 0.0f}, toRotation{0.0f, 0.0f, 0.0f};
    glm::vec3 fromScale{1.0f, 1.0f, 1.0f}, toScale{1.0f, 1.0f, 1.0f};
    float duration = 1.0f;
    float delay = 0.0f;
    bool reverse = false;
    bool loop = false;

    int size = tokens[i++].size;
    for (int j = 0; j < size; j++) {
        const jsmntok_t tok = tokens[i];
        if (compare(tok, json, "node") == 0) {
            i = parseString(tokens, i + 1, json, &nodeName);
        }
        else if (compare(tok, json, "property") == 0) {
            i = parseString(tokens, i + 1, json, &property);
        }
        else if (compare(tok, json, "fromPosition") == 0) {
            i = parseVec3(tokens, i + 1, json, &fromPosition);
        }
        else if (compare(tok, json, "toPosition") == 0) {
            i = parseVec3(tokens, i + 1, json, &toPosition);
        }
        else if (compare(tok, json, "fromRotation") == 0) {
            i = parseVec3(tokens, i + 1, json, &fromRotation);
        }
        else if (compare(tok, json, "toRotation") == 0) {
            i = parseVec3(tokens, i + 1, json, &toRotation);
        }
        else if (compare(tok, json, "fromScale") == 0) {
            i = parseVec3(tokens, i + 1, json, &fromScale);
        }
        else if (compare(tok, json, "toScale") == 0) {
            i = parseVec3(tokens, i + 1, json, &toScale);
        }
        else if (compare(tok, json, "delay") == 0) {
            i = parseFloat(tokens, i + 1, json, &delay);
        }
        else if (compare(tok, json, "duration") == 0) {
            i = parseFloat(tokens, i + 1, json, &duration);
        }
        else if (compare(tok, json, "reverse") == 0) {
            i = parseBool(tokens, i + 1, json, &reverse);
        }
        else if (compare(tok, json, "loop") == 0) {
            i = parseBool(tokens, i + 1, json, &loop);
        }
        else {
            i = parse(tokens, i + 1);
        }
        if (i < 0) {
            return i;
        }
    }

    // add animation to node
    Node* node = findNodeByName(nodeName);
    if (node != nullptr) {
        Animation* anim = (node->animation != nullptr) ? node->animation : new Animation();

        anim->addPositionKey(fromPosition, delay);
        anim->addPositionKey(toPosition, delay + duration);
        anim->setPositionProperties(reverse, loop);

        anim->addRotationKey(fromRotation, delay);
        anim->addRotationKey(toRotation, delay + duration);
        anim->setRotationProperties(reverse, loop);

        anim->addScaleKey(fromScale, delay);
        anim->addScaleKey(toScale, delay + duration);
        anim->setScaleProperties(reverse, loop);

        if (node->animation == nullptr) {
            node->animation = anim;
        }
    }
    else {
        spdlog::warn("Node not found: {}", nodeName);
    }

    return i;
}

int SceneLoader::parseAnimations(jsmntok_t* tokens, int i, const char* json, Scene& scene, PerspectiveCamera& camera) {
    CHECK_TOKTYPE(tokens[i], JSMN_ARRAY);

    int size = tokens[i++].size;
    for (int j = 0; j < size; j++) {
        i = parseAnimation(tokens, i, json, scene, camera);
    }

    return i;
}

int SceneLoader::parse(jsmntok_t* tokens, int i, const char* json, Scene& scene, PerspectiveCamera& camera) {
    CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

    int size = tokens[i++].size;
    for (int j = 0; j < size; j++) {
        const jsmntok_t tok = tokens[i];
        if (compare(tok, json, "skybox") == 0) {
            i = parseSkybox(tokens, i + 1, json, scene, camera);
        }
        else if (compare(tok, json, "materials") == 0) {
            i = parseMaterials(tokens, i + 1, json, scene, camera);
        }
        else if (compare(tok, json, "models") == 0) {
            i = parseModels(tokens, i + 1, json, scene, camera);
        }
        else if (compare(tok, json, "meshes") == 0) {
            i = parseMeshes(tokens, i + 1, json, scene, camera);
        }
        else if (compare(tok, json, "nodes") == 0) {
            i = parseNodes(tokens, i + 1, json, scene, camera);
        }
        else if (compare(tok, json, "camera") == 0) {
            i = parseCamera(tokens, i + 1, json, scene, camera);
        }
        else if (compare(tok, json, "ambientLight") == 0) {
            i = parseAmbientLight(tokens, i + 1, json, scene, camera);
        }
        else if (compare(tok, json, "directionalLight") == 0) {
            i = parseDirectionalLight(tokens, i + 1, json, scene, camera);
        }
        else if (compare(tok, json, "pointLights") == 0) {
            i = parsePointLights(tokens, i + 1, json, scene, camera);
        }
        else if (compare(tok, json, "animations") == 0) {
            i = parseAnimations(tokens, i + 1, json, scene, camera);
        }
        else {
            i = parse(tokens, i + 1);
        }
        if (i < 0) {
            return i;
        }
    }

    return i;
}

SceneLoader::~SceneLoader() {
    for (auto mesh : meshes) {
        delete mesh;
    }
    for (auto model : models) {
        delete model;
    }
    for (auto material : materials) {
        delete material;
    }
}
