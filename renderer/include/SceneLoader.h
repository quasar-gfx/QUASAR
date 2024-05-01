#ifndef SCENE_LOADER_H
#define SCENE_LOADER_H

#include <jsmn.h>

#include <Scene.h>
#include <Camera.h>

#define CHECK_TOKTYPE(tok_, type_) if ((tok_).type != (type_)) { return -1; }
#define CHECK_KEY(tok_) if ((tok_).type != JSMN_STRING || (tok_).size == 0) { return -1; }
#define STR(tok, json) std::string(json + tok.start, tok.end - tok.start)

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

class SceneLoader {
public:
    std::vector<Model*> models;
    std::vector<Mesh*> meshes;
    std::vector<PBRMaterial*> materials;

    explicit SceneLoader() = default;

    bool loadScene(std::string filename, Scene &scene, Camera &camera) {
        auto size = getFileSize(filename);
        if (size <= 0) {
            return false;
        }

        std::ifstream in(filename, std::ifstream::binary | std::ifstream::in);
        std::vector<char> json(static_cast<unsigned long>(size));
        if (!in.read(json.data(), size)) {
            return false;
        }

        JsonParser parser;
        if (!parser.readJson(json.data(), size)) {
            return false;
        }

        jsmntok_t* tokens = parser.getTokens();
        int i = parse(tokens, 0, json.data(), scene, camera);

        return i >= 0;
    }

private:
    std::ifstream::pos_type getFileSize(std::string filename) {
        std::ifstream file(filename, std::ifstream::ate | std::ifstream::binary);
        file.seekg(0, std::ios::end);
        return file.tellg();
    }

    int compare(jsmntok_t tok, const char* jsonChunk, const char* str) {
        size_t slen = strlen(str);
        size_t tlen = tok.end - tok.start;
        return (slen == tlen) ? strncmp(jsonChunk + tok.start, str, slen) : 128;
    }

    int parse(jsmntok_t const* tokens, int i) {
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

    int parseString(jsmntok_t* tokens, int i, const char* json, std::string* str) {
        CHECK_TOKTYPE(tokens[i], JSMN_STRING);
        *str = STR(tokens[i], json);
        return i + 1;
    }

    int parseBool(jsmntok_t* tokens, int i, const char* json, bool* b) {
        CHECK_TOKTYPE(tokens[i], JSMN_PRIMITIVE);
        *b = STR(tokens[i], json) == "true";
        return i + 1;
    }

    int parseFloat(jsmntok_t* tokens, int i, const char* json, float* num) {
        CHECK_TOKTYPE(tokens[i], JSMN_PRIMITIVE);
        *num = std::stof(STR(tokens[i], json));
        return i + 1;
    }

    int parseVec3(jsmntok_t* tokens, int i, const char* json, glm::vec3* vec) {
        CHECK_TOKTYPE(tokens[i], JSMN_ARRAY);

        float num;
        int size = tokens[i++].size;
        for (int j = 0; j < size; j++) {
            i = parseFloat(tokens, i, json, &num);
            (*vec)[j] = num;
        }

        return i;
    }

    int parseVec4(jsmntok_t* tokens, int i, const char* json, glm::vec4* vec) {
        CHECK_TOKTYPE(tokens[i], JSMN_ARRAY);

        float num;
        int size = tokens[i++].size;
        for (int j = 0; j < size; j++) {
            i = parseFloat(tokens, i, json, &num);
            (*vec)[j] = num;
        }

        return i;
    }

    int parseSkybox(jsmntok_t* tokens, int i, const char* json, Scene &scene, Camera &camera) {
        CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

        TextureCreateParams hdrTextureParams{
            .internalFormat = GL_RGB16F,
            .format = GL_RGB,
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
            else if (compare(tok, json, "flipped") == 0) {
                i = parseBool(tokens, i + 1, json, &hdrTextureParams.flipped);
            }
            else if (compare(tok, json, "width") == 0) {
                float width;
                i = parseFloat(tokens, i + 1, json, &width);
                skyBoxParams.width = static_cast<unsigned int>(width);
            }
            else if (compare(tok, json, "height") == 0) {
                float height;
                i = parseFloat(tokens, i + 1, json, &height);
                skyBoxParams.height = static_cast<unsigned int>(height);
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

    int parseMaterial(jsmntok_t* tokens, int i, const char* json, Scene &scene, Camera &camera) {
        CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

        std::string type;
        PBRMaterialCreateParams params{};

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
            else {
                i = parse(tokens, i + 1);
            }
            if (i < 0) {
                return i;
            }
        }

        auto material = new PBRMaterial(params);
        materials.push_back(material);

        return i;
    }

    int parseMaterials(jsmntok_t* tokens, int i, const char* json, Scene &scene, Camera &camera) {
        CHECK_TOKTYPE(tokens[i], JSMN_ARRAY);

        int size = tokens[i++].size;
        for (int j = 0; j < size; j++) {
            i = parseMaterial(tokens, i, json, scene, camera);
        }

        return i;
    }

    int parseModel(jsmntok_t* tokens, int i, const char* json, Scene &scene, Camera &camera) {
        CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

        ModelCreateParams params{};

        int size = tokens[i++].size;
        for (int j = 0; j < size; j++) {
            const jsmntok_t tok = tokens[i];
            if (compare(tok, json, "path") == 0) {
                i = parseString(tokens, i + 1, json, &params.path);
            }
            else if (compare(tok, json, "IBL") == 0) {
                i = parseBool(tokens, i + 1, json, &params.IBL);
            }
            else if (compare(tok, json, "flipTextures") == 0) {
                i = parseBool(tokens, i + 1, json, &params.flipTextures);
            }
            else if (compare(tok, json, "transparent") == 0) {
                i = parseBool(tokens, i + 1, json, &params.transparent);
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

        return i;
    }

    int parseModels(jsmntok_t* tokens, int i, const char* json, Scene &scene, Camera &camera) {
        CHECK_TOKTYPE(tokens[i], JSMN_ARRAY);

        int size = tokens[i++].size;
        for (int j = 0; j < size; j++) {
            i = parseModel(tokens, i, json, scene, camera);
        }

        return i;
    }

    int parseMesh(jsmntok_t* tokens, int i, const char* json, Scene &scene, Camera &camera) {
        CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

        std::string meshType = "cube";
        MeshCreateParams params{};

        int size = tokens[i++].size;
        for (int j = 0; j < size; j++) {
            const jsmntok_t tok = tokens[i];
            if (compare(tok, json, "type") == 0) {
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
                i = parseBool(tokens, i + 1, json, &params.IBL);
            }
            else if (compare(tok, json, "transparent") == 0) {
                i = parseBool(tokens, i + 1, json, &params.transparent);
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

        return i;
    }

    int parseMeshes(jsmntok_t* tokens, int i, const char* json, Scene &scene, Camera &camera) {
        CHECK_TOKTYPE(tokens[i], JSMN_ARRAY);

        int size = tokens[i++].size;
        for (int j = 0; j < size; j++) {
            i = parseMesh(tokens, i, json, scene, camera);
        }

        return i;
    }

    int parseNode(jsmntok_t* tokens, int i, const char* json, Scene &scene, Camera &camera) {
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
                node->setTranslation(position);
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

    int parseNodes(jsmntok_t* tokens, int i, const char* json, Scene &scene, Camera &camera) {
        CHECK_TOKTYPE(tokens[i], JSMN_ARRAY);

        int size = tokens[i++].size;
        for (int j = 0; j < size; j++) {
            i = parseNode(tokens, i, json, scene, camera);
        }

        return i;
    }

    int parseCamera(jsmntok_t* tokens, int i, const char* json, Scene &scene, Camera &camera) {
        CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

        int size = tokens[i++].size;
        for (int j = 0; j < size; j++) {
            const jsmntok_t tok = tokens[i];
            if (compare(tok, json, "fov") == 0) {
                float fov;
                i = parseFloat(tokens, i + 1, json, &fov);
                camera.fovy = glm::radians(fov);
            }
            else if (compare(tok, json, "near") == 0) {
                float near;
                i = parseFloat(tokens, i + 1, json, &near);
                camera.near = near;
            }
            else if (compare(tok, json, "far") == 0) {
                float far;
                i = parseFloat(tokens, i + 1, json, &far);
                camera.far = far;
            }
            else if (compare(tok, json, "position") == 0) {
                i = parseVec3(tokens, i + 1, json, &camera.position);
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

    int parseAmbientLight(jsmntok_t* tokens, int i, const char* json, Scene &scene, Camera &camera) {
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

    int parseDirectionalLight(jsmntok_t* tokens, int i, const char* json, Scene &scene, Camera &camera) {
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

    int parsePointLight(jsmntok_t* tokens, int i, const char* json, Scene &scene, Camera &camera) {
        CHECK_TOKTYPE(tokens[i], JSMN_OBJECT);

        PointLightCreateParams params{};

        int size = tokens[i++].size;
        for (int j = 0; j < size; j++) {
            const jsmntok_t tok = tokens[i];
            if (compare(tok, json, "color") == 0) {
                glm::vec3 color;
                i = parseVec3(tokens, i + 1, json, &color);
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

    int parsePointLights(jsmntok_t* tokens, int i, const char* json, Scene &scene, Camera &camera) {
        CHECK_TOKTYPE(tokens[i], JSMN_ARRAY);

        int size = tokens[i++].size;
        for (int j = 0; j < size; j++) {
            const jsmntok_t tok = tokens[i];

            i = parsePointLight(tokens, i, json, scene, camera);
        }

        return i;
    }

    int parse(jsmntok_t* tokens, int i, const char* json, Scene &scene, Camera &camera) {
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
            else {
                i = parse(tokens, i + 1);
            }
            if (i < 0) {
                return i;
            }
        }

        return i;
    }
};


#endif // SCENE_LOADER_H
