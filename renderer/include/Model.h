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

class Model : public Entity {
public:
    std::vector<Mesh> meshes;

    std::string rootDirectory;

    bool flipTextures = false;

    Model(const std::string &modelPath, bool flipTextures = false)
            : flipTextures(flipTextures), Entity() {
        std::cout << "Loading model: " << modelPath << std::endl;
        loadFromFile(modelPath);
    }

    void draw(Shader &shader) override;

    EntityType getType() override { return ENTITY_MESH; }

private:
    std::unordered_map<std::string, Texture> texturesLoaded;

    void loadFromFile(const std::string &path);
    void processNode(aiNode* node, const aiScene* scene);
    Mesh processMesh(aiMesh* mesh, const aiScene *scene);
    GLuint loadMaterialTexture(aiMaterial* mat, aiTextureType type);
};

#endif // MODEL_H
