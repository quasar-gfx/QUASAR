#ifndef SCENE_LOADER_H
#define SCENE_LOADER_H

#include <map>
#include <string>
#include <vector>
#include <memory>
#include <nlohmann/json.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <Path.h>
#include <Scene.h>
#include <Cameras/PerspectiveCamera.h>
#include <Materials/LitMaterial.h>
#include <Primitives/Mesh.h>
#include <Primitives/Model.h>

namespace quasar {

class SceneLoader {
public:
    SceneLoader() = default;
    ~SceneLoader();

    void clear(Scene& scene);

    void loadScene(const Path& filename, Scene& scene, PerspectiveCamera& camera);

private:
    std::vector<LitMaterial*> materials;
    std::vector<Mesh*> meshes;
    std::vector<Model*> models;

    void parse(const nlohmann::json& j, Scene& scene, PerspectiveCamera& camera);
    void parseSkybox(const nlohmann::json& j, Scene& scene);
    void parseMaterial(const nlohmann::json& j, Scene& scene);
    void parseMaterials(const nlohmann::json& j, Scene& scene);
    void parseModel(const nlohmann::json& j, Scene& scene);
    void parseModels(const nlohmann::json& j, Scene& scene);
    void parseMesh(const nlohmann::json& j, Scene& scene);
    void parseMeshes(const nlohmann::json& j, Scene& scene);
    void parseNode(const nlohmann::json& j, Scene& scene);
    void parseNodes(const nlohmann::json& j, Scene& scene);
    void parseCamera(const nlohmann::json& j, Scene& scene, PerspectiveCamera& camera);
    void parseAmbientLight(const nlohmann::json& j, Scene& scene);
    void parseDirectionalLight(const nlohmann::json& j, Scene& scene);
    void parsePointLight(const nlohmann::json& j, Scene& scene);
    void parsePointLights(const nlohmann::json& j, Scene& scene);
    void parseAnimation(const nlohmann::json& j, Scene& scene);
    void parseAnimations(const nlohmann::json& j, Scene& scene);
};

} // namespace quasar

#endif // SCENE_LOADER_H
