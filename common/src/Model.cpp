#include <fstream>
#include <sstream>
#include <iostream>

#include <Model.h>

void Model::draw(Shader &shader) {
    for (int i = 0; i < meshes.size(); i++) {
        meshes[i]->draw(shader);
    }
}

void Model::loadFromFile(const std::string &path) {
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);
    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        throw std::runtime_error("ERROR::ASSIMP:: " + std::string(importer.GetErrorString()));
    }

    rootDirectory = path.substr(0, path.find_last_of('/'));

    processNode(scene->mRootNode, scene);
}

void Model::processNode(aiNode* node, const aiScene* scene) {
    for (int i = 0; i < node->mNumMeshes; i++) {
        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
        meshes.push_back(processMesh(mesh, scene));
    }

    for (int i = 0; i < node->mNumChildren; i++) {
        processNode(node->mChildren[i], scene);
    }
}

Mesh* Model::processMesh(aiMesh* mesh, const aiScene *scene) {
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    std::vector<Texture*> textures;

    for (int i = 0; i < mesh->mNumVertices; i++) {
        Vertex vertex;
        glm::vec3 vector;

        if (mesh->HasPositions()) {
            vector.x = mesh->mVertices[i].x;
            vector.y = mesh->mVertices[i].y;
            vector.z = mesh->mVertices[i].z;
            vertex.position = vector;
        }

        if (mesh->HasNormals()) {
            vector.x = mesh->mNormals[i].x;
            vector.y = mesh->mNormals[i].y;
            vector.z = mesh->mNormals[i].z;
            vertex.normal = vector;
        }

        if (mesh->HasTextureCoords(0)) {
            glm::vec2 vec;
            vec.x = mesh->mTextureCoords[0][i].x;
            vec.y = 1.0f - mesh->mTextureCoords[0][i].y;
            vertex.texCoords = vec;
        }

        if (mesh->HasTangentsAndBitangents()) {
            vector.x = mesh->mTangents[i].x;
            vector.y = mesh->mTangents[i].y;
            vector.z = mesh->mTangents[i].z;
            vertex.tangent = vector;
        }

        vertices.push_back(vertex);
    }

    for (int i = 0; i < mesh->mNumFaces; i++) {
        aiFace face = mesh->mFaces[i];

        for (int j = 0; j < face.mNumIndices; j++) {
            indices.push_back(face.mIndices[j]);
        }
    }

    aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];

    std::vector<Texture*> diffuseMaps = loadMaterialTextures(material, aiTextureType_DIFFUSE, TEXTURE_DIFFUSE);
    textures.insert(textures.end(), diffuseMaps.begin(), diffuseMaps.end());

    std::vector<Texture*> specularMaps = loadMaterialTextures(material, aiTextureType_SPECULAR, TEXTURE_SPECULAR);
    textures.insert(textures.end(), specularMaps.begin(), specularMaps.end());

    std::vector<Texture*> normalMaps = loadMaterialTextures(material, aiTextureType_NORMALS, TEXTURE_NORMAL);
    textures.insert(textures.end(), normalMaps.begin(), normalMaps.end());

    std::vector<Texture*> heightMaps = loadMaterialTextures(material, aiTextureType_HEIGHT, TEXTURE_HEIGHT);
    textures.insert(textures.end(), heightMaps.begin(), heightMaps.end());

    return Mesh::create(vertices, indices, textures);
}

std::vector<Texture*> Model::loadMaterialTextures(aiMaterial* mat, aiTextureType type, TextureType textureType) {
    std::vector<Texture*> textures;
    for (int i = 0; i < mat->GetTextureCount(type); i++) {
        aiString str;
        mat->GetTexture(type, i, &str);

        std::string stdStr = std::string(str.C_Str());
        std::replace(stdStr.begin(), stdStr.end(), '\\', '/');

        if (texturesLoaded.count(stdStr) > 0) {
            textures.push_back(texturesLoaded[stdStr]);
        }
        else {
            Texture* texture = Texture::create(rootDirectory + '/' + stdStr, textureType);
            textures.push_back(texture);
            texturesLoaded[stdStr] = texture;
        }
    }
    return textures;
}
