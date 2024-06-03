#include <fstream>
#include <sstream>
#include <iostream>

#include <unistd.h>

#include <stb_image.h>

#include <Primatives/Model.h>
#include <assimp/pbrmaterial.h>

void Model::bindSceneAndCamera(Scene &scene, Camera &camera, glm::mat4 model, Material* overrideMaterial) {
    for (int i = 0; i < meshes.size(); i++) {
        meshes[i].bindSceneAndCamera(scene, camera, model, overrideMaterial);
    }
}

unsigned int Model::draw(Material* overrideMaterial) {
    unsigned int trianglesDrawn = 0;
    for (int i = 0; i < meshes.size(); i++) {
        trianglesDrawn += meshes[i].draw(overrideMaterial);
    }
    return trianglesDrawn;
}

void Model::loadFromFile(const ModelCreateParams &params) {
    std::string path = params.path;
    std::cout << "Loading model: " << path << std::endl;

    // use absolute path if path starts with ~/
    if (path[0] == '~') {
        char* home = getenv("HOME");
        if (home != nullptr) {
            path.replace(0, 1, home);
        }
    }

    Assimp::Importer importer;
    unsigned int flags = aiProcess_Triangulate | aiProcess_OptimizeMeshes | aiProcess_PreTransformVertices | aiProcess_CalcTangentSpace | aiProcess_FlipUVs;
    scene = importer.ReadFile(path, flags);
    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        throw std::runtime_error("ERROR::ASSIMP:: " + std::string(importer.GetErrorString()));
    }

    rootDirectory = path.substr(0, path.find_last_of('/'))  + '/';

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

    // set up vertices
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

            vector.x = mesh->mBitangents[i].x;
            vector.y = mesh->mBitangents[i].y;
            vector.z = mesh->mBitangents[i].z;
            vertex.bitangent = vector;
        }

        vertices.push_back(vertex);
    }

    // set up indices
    for (int i = 0; i < mesh->mNumFaces; i++) {
        aiFace face = mesh->mFaces[i];

        for (int j = 0; j < face.mNumIndices; j++) {
            indices.push_back(face.mIndices[j]);
        }
    }

    // set up material
    aiMaterial* aiMat = scene->mMaterials[mesh->mMaterialIndex];

    MeshCreateParams meshParams{};
    if (material != nullptr) {
        meshParams.material = material;
    }
    else {
        PBRMaterialCreateParams materialParams{};

        aiColor3D color;
        if (aiMat->Get(AI_MATKEY_COLOR_DIFFUSE, color) == AI_SUCCESS) {
            materialParams.color = glm::vec3(color.r, color.g, color.b);
        }

        aiString alphaMode;
        if (aiMat->Get(AI_MATKEY_GLTF_ALPHAMODE, alphaMode) == AI_SUCCESS) {
            if (alphaMode == aiString("BLEND")) {
                materialParams.transparent = true;
            }
        }
        float opacity = 1.0;
        if (aiMat->Get(AI_MATKEY_OPACITY, opacity) == AI_SUCCESS) {
            if (opacity < 1.0) {
                materialParams.transparent = true;
            }
            if (opacity <= 0.0f) opacity = 1.0f;
            materialParams.opacity = opacity;
        }

        float shininess;
        if (aiMat->Get(AI_MATKEY_SHININESS, shininess) == AI_SUCCESS) {
            // convert shininess to roughness
            float roughness = sqrt(2.0f / (shininess + 2.0f));
            materialParams.roughness = roughness;
        }

        float metallic;
        if (aiMat->Get(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_METALLIC_FACTOR, metallic) == AI_SUCCESS) {
            materialParams.metallic = metallic;
        }

        TextureID diffuseMap = loadMaterialTexture(aiMat, aiTextureType_DIFFUSE);
        TextureID normalMap = loadMaterialTexture(aiMat, aiTextureType_NORMALS);
        TextureID metallicMap = loadMaterialTexture(aiMat, aiTextureType_METALNESS);
        TextureID roughnessMap = loadMaterialTexture(aiMat, aiTextureType_DIFFUSE_ROUGHNESS);
        // sometimes gltf models have metal and roughness in a single texture
        if (metallicMap == 0 || roughnessMap == 0) {
            // the metallic-roughness texture is sometimes stored in aiTextureType_UNKNOWN
            metallicMap = loadMaterialTexture(aiMat, aiTextureType_UNKNOWN);
            materialParams.metalRoughnessCombined = true;
        }
        TextureID aoMap = loadMaterialTexture(aiMat, aiTextureType_AMBIENT_OCCLUSION);

        textures = {diffuseMap, normalMap, metallicMap, roughnessMap, aoMap};
        materialParams.albedoTextureID = diffuseMap;
        materialParams.normalTextureID = normalMap;
        materialParams.metallicTextureID = metallicMap;
        materialParams.roughnessTextureID = roughnessMap;
        materialParams.aoTextureID = aoMap;

        meshParams.material = new PBRMaterial(materialParams);
    }

    meshParams.vertices = vertices;
    meshParams.indices = indices;
    meshParams.wireframe = wireframe;
    meshParams.pointcloud = pointcloud;
    meshParams.IBL = IBL;

    return Mesh(meshParams);
}

TextureID Model::loadMaterialTexture(aiMaterial* mat, aiTextureType type) {
    // if the texture type doesn't exist, return 0
    if (mat->GetTextureCount(type) == 0) {
        return 0;
    }

    // else if the texture type exists, load the texture
    aiString aiTexturePath;
    mat->GetTexture(type, 0, &aiTexturePath); // only grab the first texture of each type

    std::string texturePath = rootDirectory;
    texturePath = texturePath.append(aiTexturePath.C_Str());
    std::replace(texturePath.begin(), texturePath.end(), '\\', '/');

    // if we've loaded this texture already, return the already loaded texture
    if (texturesLoaded.count(texturePath) > 0) {
        return texturesLoaded[texturePath].ID;
    }

    // if texture is embedded into the file, read it from memory
    if (aiTexturePath.length > 0 && aiTexturePath.data[0] == '*') { // seems like assimp uses * as a prefix for embedded textures
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

                stbi_image_free(data);
                texturesLoaded[texturePath] = texture;
                return texturesLoaded[texturePath].ID;
            }
        }

        return 0;
    }
    // else load the texture from external file
    else {
        Texture texture = Texture({
            .wrapS = GL_REPEAT,
            .wrapT = GL_REPEAT,
            .minFilter = GL_LINEAR_MIPMAP_LINEAR,
            .magFilter = GL_LINEAR,
            .gammaCorrected = gammaCorrected,
            .path = texturePath
        });
        texturesLoaded[texturePath] = texture;
        return texturesLoaded[texturePath].ID;
    }
}
