#include <fstream>
#include <sstream>
#include <iostream>

#include <Model.h>

void Model::draw(Shader &shader) {
    for (int i = 0; i < meshes.size(); i++) {
        meshes[i].draw(shader);
    }
}

void Model::loadFromFile(const std::string &path) {
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_OptimizeMeshes | aiProcess_CalcTangentSpace | aiProcess_FlipUVs);
    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        throw std::runtime_error("ERROR::ASSIMP:: " + std::string(importer.GetErrorString()));
    }

    rootDirectory = path.substr(0, path.find_last_of('/'))  + '/';

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

Mesh Model::processMesh(aiMesh* mesh, const aiScene* scene) {
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    std::vector<GLuint> textures;

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
            if (flipTextures) {
                vec.y = 1.0f - mesh->mTextureCoords[0][i].y;
            }
            else {
                vec.y = mesh->mTextureCoords[0][i].y;
            }
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

    textures = {
        loadMaterialTexture(material, aiTextureType_DIFFUSE),
        loadMaterialTexture(material, aiTextureType_SPECULAR),
        loadMaterialTexture(material, aiTextureType_NORMALS),
        loadMaterialTexture(material, aiTextureType_HEIGHT)
    };

    float shininess = 0.0f;
    aiGetMaterialFloat(material, AI_MATKEY_SHININESS, &shininess);

    return Mesh(vertices, indices, textures, shininess);
}

GLuint Model::loadMaterialTexture(aiMaterial* mat, aiTextureType type) {
    if (mat->GetTextureCount(type) > 0) {
        aiString str;
        mat->GetTexture(type, 0, &str);

        std::string texturePath = rootDirectory;
        texturePath = texturePath.append(str.C_Str());

        std::replace(texturePath.begin(), texturePath.end(), '\\', '/');

        if (texturesLoaded.count(texturePath) == 0) {
            Texture texture = Texture(texturePath);
            texturesLoaded[texturePath] = texture;
        }

        return texturesLoaded[texturePath].ID;
    }
    else {
        return 0;
    }
}
