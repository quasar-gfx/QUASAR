#include <fstream>
#include <sstream>
#include <iostream>

#include <stb_image.h>

#include <Primatives/Model.h>

void Model::bindSceneAndCamera(Scene& scene, Camera& camera, glm::mat4 model, Material* overrideMaterial) {
    for (int i = 0; i < meshes.size(); i++) {
        meshes[i].bindSceneAndCamera(scene, camera, model, overrideMaterial);
    }
}

void Model::draw(Material* overrideMaterial) {
    for (int i = 0; i < meshes.size(); i++) {
        meshes[i].draw(overrideMaterial);
    }
}

void Model::loadFromFile(const ModelCreateParams &params) {
    Assimp::Importer importer;
    unsigned int flags = aiProcess_Triangulate | aiProcess_OptimizeMeshes | aiProcess_PreTransformVertices | aiProcess_CalcTangentSpace | aiProcess_FlipUVs;
    scene = importer.ReadFile(params.path, flags);
    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        throw std::runtime_error("ERROR::ASSIMP:: " + std::string(importer.GetErrorString()));
    }

    rootDirectory = params.path.substr(0, params.path.find_last_of('/'))  + '/';

    processNode(scene->mRootNode, scene, params.material);
}

void Model::processNode(aiNode* node, const aiScene* scene, PBRMaterial* material) {
    for (int i = 0; i < node->mNumMeshes; i++) {
        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
        meshes.push_back(processMesh(mesh, scene, material));
    }

    for (int i = 0; i < node->mNumChildren; i++) {
        processNode(node->mChildren[i], scene, material);
    }
}

Mesh Model::processMesh(aiMesh* mesh, const aiScene* scene, PBRMaterial* material) {
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

    aiMaterial* aiMat = scene->mMaterials[mesh->mMaterialIndex];

    if (material != nullptr) {
        MeshCreateParams meshParams{
            .vertices = vertices,
            .indices = indices,
            .material = material,
            .wireframe = wireframe,
            .pointcloud = pointcloud
        };
        return Mesh(meshParams);
    }
    else {
        textures = {
            loadMaterialTexture(aiMat, aiTextureType_DIFFUSE),
            loadMaterialTexture(aiMat, aiTextureType_NORMALS),
            loadMaterialTexture(aiMat, aiTextureType_METALNESS),
            loadMaterialTexture(aiMat, aiTextureType_DIFFUSE_ROUGHNESS),
            loadMaterialTexture(aiMat, aiTextureType_AMBIENT_OCCLUSION)
        };

        return Mesh({
            .vertices = vertices,
            .indices = indices,
            .material = new PBRMaterial({
                .albedoTextureID = textures[0],
                .normalTextureID = textures[1],
                .metallicTextureID = textures[2],
                .roughnessTextureID = textures[3],
                .aoTextureID = textures[4]
            }),
            .wireframe = wireframe,
            .pointcloud = pointcloud
        });
    }
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
                        GLint internalFormat;
                        GLenum format;
                        if (texChannels == 1) {
                            internalFormat = GL_RED;
                            format = GL_RED;
                        }
                        else if (texChannels == 3) {
                            internalFormat = gammaCorrected ? GL_SRGB : GL_RGB;
                            format = GL_RGB;
                        }
                        else if (texChannels == 4) {
                            internalFormat = gammaCorrected ? GL_SRGB_ALPHA : GL_RGBA;
                            format = GL_RGBA;
                        }

                        Texture texture = Texture({
                            .width = static_cast<unsigned int>(texWidth),
                            .height = static_cast<unsigned int>(texHeight),
                            .internalFormat = internalFormat,
                            .format = format,
                            .wrapS = GL_REPEAT,
                            .wrapT = GL_REPEAT,
                            .minFilter = GL_LINEAR_MIPMAP_LINEAR,
                            .magFilter = GL_LINEAR,
                            .data = data
                        });
                        glGenerateMipmap(GL_TEXTURE_2D);

                        stbi_image_free(data);
                        texturesLoaded[texturePath] = texture;
                        return texturesLoaded[texturePath].ID;
                    }
                }

                return 0;
            }
            else {
                Texture texture = Texture({
                    .wrapS = GL_REPEAT,
                    .wrapT = GL_REPEAT,
                    .minFilter = GL_LINEAR_MIPMAP_LINEAR,
                    .magFilter = GL_LINEAR,
                    .gammaCorrected = (type == aiTextureType_DIFFUSE) ? gammaCorrected : false,
                    .path = texturePath
                });
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
