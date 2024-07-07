#ifndef MODEL_H
#define MODEL_H

#include <string>
#include <unordered_map>
#include <vector>

#include <glad/glad.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <Vertex.h>
#include <Shaders/Shader.h>
#include <Primatives/Mesh.h>
#include <Materials/PBRMaterial.h>
#include <Primatives/Entity.h>

struct ModelCreateParams {
    bool flipTextures = false;
    bool wireframe = false;
    bool pointcloud = false;
    bool visible = true;
    bool gammaCorrected = false;
    float IBL = 1.0;
    PBRMaterial* material;
    std::string path;
};

class Model : public Entity {
public:
    std::vector<Mesh*> meshes;

    std::string rootDirectory;

    bool flipTextures = false;
    bool wireframe = false;
    bool pointcloud = false;
    bool visible = true;
    bool gammaCorrected = false;
    float IBL = 1.0;

    explicit Model(const ModelCreateParams &params)
            : flipTextures(params.flipTextures)
            , wireframe(params.wireframe)
            , pointcloud(params.pointcloud)
            , visible(params.visible)
            , gammaCorrected(params.gammaCorrected)
            , IBL(params.IBL)
            , Entity() {
        loadFromFile(params);
    }
    ~Model();

    void bindSceneAndCamera(const Scene &scene, const Camera &camera, const glm::mat4 &model, const Material* overrideMaterial = nullptr) override;
    unsigned int draw(const Scene &scene, const Camera &camera, const glm::mat4 &model, bool frustumCull, const Material* overrideMaterial) override;
    unsigned int draw(const Scene &scene, const Camera &camera, const glm::mat4 &model, const BoundingSphere &boundingSphere, const Material* overrideMaterial) override;

    EntityType getType() const override { return EntityType::MESH; }

private:
    const aiScene* scene;
    std::unordered_map<std::string, Texture> texturesLoaded;

    void loadFromFile(const ModelCreateParams &params);
    void processNode(aiNode* node, const aiScene* scene, PBRMaterial* material);
    Mesh* processMesh(aiMesh* mesh, const aiScene *scene, PBRMaterial* material);
    TextureID loadMaterialTexture(aiMaterial const* mat, aiTextureType type);
    int32_t getEmbeddedTextureId(const aiString& path);
};

#endif // MODEL_H
