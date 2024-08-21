#ifndef SCENE_LOADER_H
#define SCENE_LOADER_H

#define JSMN_STATIC
#include <jsmn.h>

#include <fstream>

#include <Scene.h>
#include <PerspectiveCamera.h>
#include <Primatives/Primatives.h>

#define CHECK_TOKTYPE(tok_, type_) if ((tok_).type != (type_)) { return -1; }
#define CHECK_KEY(tok_) if ((tok_).type != JSMN_STRING || (tok_).size == 0) { return -1; }
#define STR(tok, json) std::string(json + tok.start, tok.end - tok.start)

class SceneLoader {
public:
    std::vector<Model*> models;
    std::vector<Mesh*> meshes;
    std::vector<PBRMaterial*> materials;

    SceneLoader() = default;
    ~SceneLoader();

    void loadScene(const std::string &filename, Scene &scene, PerspectiveCamera &camera);
    void clearScene(Scene &scene, PerspectiveCamera &camera);

private:
    int compare(jsmntok_t tok, const char* jsonChunk, const char* str);

    int parse(jsmntok_t const* tokens, int i);
    int parseString(jsmntok_t* tokens, int i, const char* json, std::string* str);
    int parseBool(jsmntok_t* tokens, int i, const char* json, bool* b);
    int parseFloat(jsmntok_t* tokens, int i, const char* json, float* num);
    int parseVec3(jsmntok_t* tokens, int i, const char* json, glm::vec3* vec);
    int parseVec4(jsmntok_t* tokens, int i, const char* json, glm::vec4* vec);
    int parseSkybox(jsmntok_t* tokens, int i, const char* json, Scene &scene, PerspectiveCamera &camera);
    int parseMaterial(jsmntok_t* tokens, int i, const char* json, Scene &scene, PerspectiveCamera &camera);
    int parseMaterials(jsmntok_t* tokens, int i, const char* json, Scene &scene, PerspectiveCamera &camera);
    int parseModel(jsmntok_t* tokens, int i, const char* json, Scene &scene, PerspectiveCamera &camera);
    int parseModels(jsmntok_t* tokens, int i, const char* json, Scene &scene, PerspectiveCamera &camera);
    int parseMesh(jsmntok_t* tokens, int i, const char* json, Scene &scene, PerspectiveCamera &camera);
    int parseMeshes(jsmntok_t* tokens, int i, const char* json, Scene &scene, PerspectiveCamera &camera);
    int parseNode(jsmntok_t* tokens, int i, const char* json, Scene &scene, PerspectiveCamera &camera);
    int parseNodes(jsmntok_t* tokens, int i, const char* json, Scene &scene, PerspectiveCamera &camera);
    int parseCamera(jsmntok_t* tokens, int i, const char* json, Scene &scene, PerspectiveCamera &camera);
    int parseAmbientLight(jsmntok_t* tokens, int i, const char* json, Scene &scene, PerspectiveCamera &camera);
    int parseDirectionalLight(jsmntok_t* tokens, int i, const char* json, Scene &scene, PerspectiveCamera &camera);
    int parsePointLight(jsmntok_t* tokens, int i, const char* json, Scene &scene, PerspectiveCamera &camera);
    int parsePointLights(jsmntok_t* tokens, int i, const char* json, Scene &scene, PerspectiveCamera &camera);
    int parse(jsmntok_t* tokens, int i, const char* json, Scene &scene, PerspectiveCamera &camera);
};


#endif // SCENE_LOADER_H
