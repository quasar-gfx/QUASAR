#ifndef MODEL_H
#define MODEL_H

#include <glad/glad.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <Shader.h>
#include <Mesh.h>

#include <string>
#include <unordered_map>
#include <vector>

class Model {
public:
    std::vector<Mesh*> meshes;

    std::string rootDirectory;

    void draw(Shader &shader);

    static Model* create(const std::string &modelPath) {
        return new Model(modelPath);
    }

private:
    std::unordered_map<std::string, Texture*> texturesLoaded;

    Model(const std::string &modelPath) {
        std::cout << "Loading model: " << modelPath << std::endl;
        loadFromFile(modelPath);
    }

    void loadFromFile(const std::string &path);
    void processNode(aiNode* node, const aiScene* scene);
    Mesh* processMesh(aiMesh* mesh, const aiScene *scene);
    std::vector<Texture*> loadMaterialTextures(aiMaterial* mat, aiTextureType type, TextureType textureType);
};

#endif // MODEL_H
