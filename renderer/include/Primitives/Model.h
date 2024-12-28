#ifndef MODEL_H
#define MODEL_H

#include <string>
#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <assimp/scene.h>

#include <Vertex.h>
#include <Shaders/Shader.h>
#include <Primitives/Mesh.h>
#include <Materials/PBRMaterial.h>
#include <Primitives/Entity.h>

struct ModelCreateParams {
    bool flipTextures = false;
    bool gammaCorrected = false;
    float IBL = 1.0;
    PBRMaterial* material;
    std::string path;
};

class Model : public Entity {
public:
    std::vector<Mesh*> meshes;
    Node rootNode;

    std::string rootDirectory;

    bool flipTextures = false;
    bool gammaCorrected = false;
    float IBL = 1.0;

    PBRMaterial* material;

    bool isGLTF = false;

    Model(const ModelCreateParams &params)
            : flipTextures(params.flipTextures)
            , gammaCorrected(params.gammaCorrected)
            , IBL(params.IBL) {
        loadFromFile(params);
    }
    ~Model();

    virtual void bindMaterial(const Scene &scene, const glm::mat4 &model,
                              const Material* overrideMaterial = nullptr, const Texture* prevDepthMap = nullptr) override;

    virtual RenderStats draw(GLenum primativeType, const Camera &camera, const glm::mat4 &model,
                             bool frustumCull = true, const Material* overrideMaterial = nullptr) override;
    virtual RenderStats draw(GLenum primativeType, const Camera &camera, const glm::mat4 &model,
                             const BoundingSphere &boundingSphere, const Material* overrideMaterial = nullptr) override;
    virtual void updateAnimations(float dt) override;

    EntityType getType() const override { return EntityType::MODEL; }

    Node* findNodeByName(const std::string &name);

private:
    const aiScene* scene;
    std::unordered_map<std::string, Texture*> texturesLoaded;

    void loadFromFile(const ModelCreateParams &params);
    void processNode(aiNode* aiNode, const aiScene* scene, Node* node, PBRMaterial* material);
    Mesh* processMesh(aiMesh* mesh, const aiScene* scene, PBRMaterial* material);
    void processMaterial(aiMaterial const* aiMat, PBRMaterialCreateParams &materialParams);
    Texture* loadMaterialTexture(aiMaterial const* aiMat, aiString aiTexturePath, bool shouldGammaCorrect = false);
    int32_t getEmbeddedTextureId(const aiString &path);

    RenderStats drawNode(const Node* node,
                         GLenum primativeType, const Camera &camera,
                         const glm::mat4& parentTransform, const glm::mat4 &model,
                         bool frustumCull = true, const Material* overrideMaterial = nullptr);
    RenderStats drawNode(const Node* node,
                         GLenum primativeType, const Camera &camera,
                         const glm::mat4& parentTransform, const glm::mat4 &model,
                         const BoundingSphere &boundingSphere, const Material* overrideMaterial = nullptr);
};

#endif // MODEL_H
