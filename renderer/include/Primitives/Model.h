#ifndef MODEL_H
#define MODEL_H

#include <string>
#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <assimp/scene.h>

#include <Primitives/Node.h>
#include <Primitives/Mesh.h>
#include <Materials/LitMaterial.h>

namespace quasar {

struct ModelCreateParams {
    bool flipTextureY = true;
    bool gammaCorrected = false;
    float IBL = 1.0;
    std::string path;
};

class Model : public Node {
public:
    std::string rootDirectory;

    bool gammaCorrected;
    float IBL;

    const LitMaterial* material;

    Model(const ModelCreateParams& params);
    ~Model();

private:
    bool flipTextureY;

    std::vector<Mesh*> meshes;
    std::vector<const LitMaterial*> materials;

    const aiScene* scene;

    // One image can back both a color slot and a data slot, so how it is decoded is part
    // of its identity: keying on the path alone would hand back the wrong internal format
    struct TextureKey {
        std::string path;
        bool gammaCorrected = false;

        bool operator==(const TextureKey& other) const {
            return gammaCorrected == other.gammaCorrected && path == other.path;
        }
    };
    struct TextureKeyHash {
        size_t operator()(const TextureKey& key) const {
            return std::hash<std::string>{}(key.path) ^ (static_cast<size_t>(key.gammaCorrected) << 1);
        }
    };

    std::unordered_map<TextureKey, Texture*, TextureKeyHash> texturesCache;

    void loadFromFile(const ModelCreateParams& params);
    void processNode(aiNode* aiNode, const aiScene* scene, Node* node);
    Mesh* processMesh(aiMesh* mesh, const aiScene* scene);
    void processAnimations(const aiScene* scene);
    void processMaterial(aiMaterial const* aiMat, LitMaterialCreateParams& materialParams);
    Texture* loadMaterialTexture(aiMaterial const* aiMat, aiString aiTexturePath, bool shouldGammaCorrect = false);
    int32_t getEmbeddedTextureId(const aiString& path);
};

} // namespace quasar

#endif // MODEL_H
