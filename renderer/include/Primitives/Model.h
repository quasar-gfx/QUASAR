#ifndef MODEL_H
#define MODEL_H

#include <string>
#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <assimp/scene.h>

#include <Vertex.h>
#include <Primitives/Node.h>
#include <Primitives/Mesh.h>
#include <Materials/LitMaterial.h>

namespace quasar {

struct ModelCreateParams {
    bool flipTextures = false;
    bool gammaCorrected = false;
    float IBL = 1.0;
    const LitMaterial* material;
    std::string path;
};

class Model : public Node {
public:
    bool isGLTF = false;

    std::string rootDirectory;

    bool gammaCorrected;
    float IBL;

    const LitMaterial* material;

    Model(const ModelCreateParams& params);
    ~Model();

private:
    bool flipTextures;

    std::vector<Mesh*> meshes;

    const aiScene* scene;
    std::unordered_map<std::string, Texture*> texturesCache;

    void loadFromFile(const ModelCreateParams& params);
    void processNode(aiNode* aiNode, const aiScene* scene, Node* node, const LitMaterial* material);
    Mesh* processMesh(aiMesh* mesh, const aiScene* scene, const LitMaterial* material);
    void processAnimations(const aiScene* scene);
    void processMaterial(aiMaterial const* aiMat, LitMaterialCreateParams& materialParams);
    Texture* loadMaterialTexture(aiMaterial const* aiMat, aiString aiTexturePath, bool shouldGammaCorrect = false);
    int32_t getEmbeddedTextureId(const aiString& path);
};

} // namespace quasar

#endif // MODEL_H
