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

#include <Shader.h>
#include <Mesh.h>
#include <Entity.h>

struct ModelCreateParams {
    std::string path;
    std::vector<TextureID> inputTextures;
    bool flipTextures = false;
    bool wireframe = false;
    bool drawAsPointCloud = false;
    bool gammaCorrected = false;
};

class Model : public Entity {
public:
    std::vector<Mesh> meshes;

    std::string rootDirectory;

    bool flipTextures = false;

    bool wireframe = false;
    bool drawAsPointCloud = false;
    bool gammaCorrected = false;

    Model(const ModelCreateParams &params)
            : flipTextures(params.flipTextures),
                wireframe(params.wireframe), drawAsPointCloud(params.drawAsPointCloud),
                gammaCorrected(params.gammaCorrected),
                Entity() {
        std::cout << "Loading model: " << params.path << std::endl;
        loadFromFile(params);
    }

    void draw(Shader &shader) override;

    EntityType getType() override { return ENTITY_MESH; }

private:
    const aiScene* scene;
    std::unordered_map<std::string, Texture> texturesLoaded;

    void loadFromFile(const ModelCreateParams &params);
    void processNode(aiNode* node, const aiScene* scene, std::vector<TextureID> inputTextures);
    Mesh processMesh(aiMesh* mesh, const aiScene *scene, std::vector<TextureID> inputTextures);
    GLuint loadMaterialTexture(aiMaterial* mat, aiTextureType type);
};

#endif // MODEL_H
