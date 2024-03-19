#include <fstream>
#include <sstream>
#include <iostream>

#include <stb_image.h>

#include <Model.h>

void Model::draw(Shader &shader) {
    for (int i = 0; i < meshes.size(); i++) {
        meshes[i].draw(shader);
    }
}

void Model::loadFromFile(const std::string &path, std::vector<TextureID> inputTextures) {
    Assimp::Importer importer;
    unsigned int flags = aiProcess_Triangulate | aiProcess_OptimizeMeshes | aiProcess_PreTransformVertices | aiProcess_CalcTangentSpace | aiProcess_FlipUVs;
    scene = importer.ReadFile(path, flags);
    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        throw std::runtime_error("ERROR::ASSIMP:: " + std::string(importer.GetErrorString()));
    }

    rootDirectory = path.substr(0, path.find_last_of('/'))  + '/';

    processNode(scene->mRootNode, scene, inputTextures);
}

void Model::processNode(aiNode* node, const aiScene* scene, std::vector<TextureID> inputTextures) {
    for (int i = 0; i < node->mNumMeshes; i++) {
        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
        meshes.push_back(processMesh(mesh, scene, inputTextures));
    }

    for (int i = 0; i < node->mNumChildren; i++) {
        processNode(node->mChildren[i], scene, inputTextures);
    }
}

Mesh Model::processMesh(aiMesh* mesh, const aiScene* scene, std::vector<TextureID> inputTextures) {
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    std::vector<TextureID> textures;

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

    if (inputTextures.size() > 0) {
        for (int i = 0; i < Mesh::numTextures; i++) {
            if (i < inputTextures.size()) {
                textures.push_back(inputTextures[i]);
            }
            else {
                textures.push_back(0);
            }
        }
    }
    else {
        textures = {
            loadMaterialTexture(material, aiTextureType_DIFFUSE),
            loadMaterialTexture(material, aiTextureType_SPECULAR),
            loadMaterialTexture(material, aiTextureType_NORMALS),
            loadMaterialTexture(material, aiTextureType_METALNESS),
            loadMaterialTexture(material, aiTextureType_DIFFUSE_ROUGHNESS),
            loadMaterialTexture(material, aiTextureType_AMBIENT_OCCLUSION)
        };
    }

    float shininess = 0.0f;
    aiGetMaterialFloat(material, AI_MATKEY_SHININESS, &shininess);

    auto res = Mesh(vertices, indices, textures, shininess);
    res.wireframe = wireframe;
    return res;
}

GLuint Model::loadMaterialTexture(aiMaterial* mat, aiTextureType type) {
    // if the texture type exists, load the texture
    if (mat->GetTextureCount(type) > 0) {
        aiString aiTexturePath;
        mat->GetTexture(type, 0, &aiTexturePath); // only grab the first texture of each type

        std::string texturePath = rootDirectory;
        texturePath = texturePath.append(aiTexturePath.C_Str());
        std::replace(texturePath.begin(), texturePath.end(), '\\', '/');

        // if we havent loaded this texture yet
        if (texturesLoaded.count(texturePath) == 0) {
            // seems like assimp uses * as a prefix for embedded textures
            if (aiTexturePath.length > 0 && aiTexturePath.data[0] == '*') {
                const aiTexture* aiTexture = scene->GetEmbeddedTexture(aiTexturePath.C_Str());
                if (aiTexture) {
                    int texWidth, texHeight, texChannels;
                    unsigned char* data = stbi_load_from_memory(reinterpret_cast<unsigned char*>(aiTexture->pcData),
                                                                aiTexture->mWidth, &texWidth, &texHeight, &texChannels, 0);
                    if (data) {
                        GLenum internalFormat, format;
                        if (texChannels == 1) {
                            internalFormat = GL_RED;
                            format = GL_RED;
                        }
                        else if (texChannels == 3) {
                            internalFormat = GL_RGB;
                            format = GL_RGB;
                        }
                        else if (texChannels == 4) {
                            internalFormat = GL_RGBA;
                            format = GL_RGBA;
                        }

                        Texture texture = Texture(texWidth, texHeight, internalFormat, format, GL_UNSIGNED_BYTE, GL_REPEAT, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR, data);
                        glGenerateMipmap(GL_TEXTURE_2D);

                        stbi_image_free(data);
                        texturesLoaded[texturePath] = texture;
                        return texturesLoaded[texturePath].ID;
                    }
                }

                return 0;
            }
            else {
                Texture texture = Texture(texturePath);
                texturesLoaded[texturePath] = texture;
                return texturesLoaded[texturePath].ID;
            }
        }

        return 0;
    }
    // else return 0, indicating that there is no texture
    else {
        return 0;
    }
}
