// adpated from: https://github.com/google/filament/blob/main/libs/filamentapp/src/MeshAssimp.cpp
#include <iostream>

#include <unistd.h>

#include <stb_image.h>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/cimport.h>
#include <assimp/GltfMaterial.h>

#include <Primatives/Model.h>

void Model::bindMaterial(const Scene &scene, const Camera &camera, const glm::mat4 &model, const Material* overrideMaterial) {
    for (auto& mesh : meshes) {
        mesh->bindMaterial(scene, camera, model, overrideMaterial);
    }
}

unsigned int Model::draw(const Scene &scene, const Camera &camera, const glm::mat4 &model, bool frustumCull, const Material* overrideMaterial) {
    unsigned int trianglesDrawn = 0;

    for (auto& mesh : meshes) {
        trianglesDrawn += mesh->draw(scene, camera, model, frustumCull, overrideMaterial);
    }

    return trianglesDrawn;
}

unsigned int Model::draw(const Scene &scene, const Camera &camera, const glm::mat4 &model, const BoundingSphere &boundingSphere, const Material* overrideMaterial) {
    unsigned int trianglesDrawn = 0;

    for (auto& mesh : meshes) {
        trianglesDrawn += mesh->draw(scene, camera, model, boundingSphere, overrideMaterial);
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
    importer.SetPropertyInteger(AI_CONFIG_PP_SBP_REMOVE, aiPrimitiveType_LINE | aiPrimitiveType_POINT);
    importer.SetPropertyBool(AI_CONFIG_IMPORT_COLLADA_IGNORE_UP_DIRECTION, true);
    importer.SetPropertyBool(AI_CONFIG_PP_PTV_KEEP_HIERARCHY, true);

    unsigned int flags = \
            // normals and tangents
            aiProcess_GenSmoothNormals |
            aiProcess_CalcTangentSpace |
            // UV Coordinates
            aiProcess_GenUVCoords |
            // topology optimization
            aiProcess_FindInstances |
            aiProcess_OptimizeMeshes |
            aiProcess_JoinIdenticalVertices |
            // misc optimization
            aiProcess_ImproveCacheLocality |
            aiProcess_SortByPType |
            // we only support triangles
            aiProcess_Triangulate |
            // pre-transform vertices
            aiProcess_PreTransformVertices;
    scene = importer.ReadFile(path, flags);
    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        throw std::runtime_error("ERROR::ASSIMP:: " + std::string(importer.GetErrorString()));
    }

    std::string extension = path.substr(path.find_last_of('.') + 1);
    size_t index = importer.GetImporterIndex(extension.c_str());
    const aiImporterDesc* importerDesc = importer.GetImporterInfo(index);
    isGLTF = importerDesc &&
            (!strncmp("glTF Importer",  importerDesc->mName, 13) ||
             !strncmp("glTF2 Importer", importerDesc->mName, 14));

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

Mesh* Model::processMesh(aiMesh* mesh, const aiScene* scene, PBRMaterial* material) {
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;

    // set up vertices
    glm::vec3 min = glm::vec3(FLT_MAX);
    glm::vec3 max = glm::vec3(-FLT_MAX);
    for (int i = 0; i < mesh->mNumVertices; i++) {
        Vertex vertex;
        glm::vec3 vector;

        if (mesh->HasPositions()) {
            vector.x = mesh->mVertices[i].x;
            vector.y = mesh->mVertices[i].y;
            vector.z = mesh->mVertices[i].z;
            vertex.position = vector;

            min = glm::min(min, vector);
            max = glm::max(max, vector);
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
        else {
            vertex.bitangent = glm::normalize(glm::cross(vertex.normal, glm::vec3(1.0f, 0.0f, 0.0f)));
            vertex.tangent = glm::normalize(glm::cross(vertex.normal, vertex.bitangent));
        }

        vertices.push_back(vertex);
    }

    // set up AABB
    aabb.update(min, max);

    // set up indices
    for (int i = 0; i < mesh->mNumFaces; i++) {
        aiFace face = mesh->mFaces[i];

        for (int j = 0; j < face.mNumIndices; j++) {
            indices.push_back(face.mIndices[j]);
        }
    }

    // set up material
    uint32_t materialId = mesh->mMaterialIndex;
    aiMaterial const* aiMat = scene->mMaterials[materialId];

    MeshCreateParams meshParams{};
    if (material != nullptr) {
        meshParams.material = material;
    }
    else {
        PBRMaterialCreateParams materialParams{};
        processMaterial(aiMat, materialParams);
        meshParams.material = new PBRMaterial(materialParams);
    }

    meshParams.vertices = vertices;
    meshParams.indices = indices;
    meshParams.wireframe = wireframe;
    meshParams.pointcloud = pointcloud;
    meshParams.IBL = IBL;

    return new Mesh(meshParams);
}

void Model::processMaterial(const aiMaterial* aiMat, PBRMaterialCreateParams &materialParams) {
    aiString alphaMode;
    aiString baseColorPath;
    aiString normalPath;
    aiString AOPath;
    aiString MPath, RPath, MRPath;
    aiString emissivePath;
    aiTextureMapMode mapMode[3];

    aiColor4D baseColorFactor;
    aiColor3D emissiveFactor;
    float metallicFactor = 1.0;
    float roughnessFactor = 1.0;

    aiColor3D color;
    glm::vec4 baseColor = glm::vec4(1.0f);
    if (aiMat->Get(AI_MATKEY_COLOR_DIFFUSE, color) == AI_SUCCESS) {
        baseColor = glm::vec4(color.r, color.g, color.b, baseColor.a);
    }

    float opacity;
    if (aiMat->Get(AI_MATKEY_OPACITY, opacity) != AI_SUCCESS) {
        opacity = 1.0f;
    }
    if (opacity <= 0.0f) opacity = 1.0f;
    baseColor.a = opacity;

    float shininess;
    if (aiMat->Get(AI_MATKEY_SHININESS, shininess) != AI_SUCCESS) {
        shininess = 0.0f;
    }

    // convert shininess to roughness
    float roughness = sqrt(2.0f / (shininess + 2.0f));
    materialParams.roughness = roughness;

    materialParams.metallic = 0.0f;
    if (aiMat->Get(AI_MATKEY_COLOR_SPECULAR, color) == AI_SUCCESS) {
        // if there's a non-grey specular color, assume a metallic surface
        if (color.r != color.g && color.r != color.b) {
            materialParams.metallic = 1.0f;
            baseColor = glm::vec4(color.r, color.g, color.b, baseColor.a);
        }
        else {
            if (baseColor.r == 0.0f && baseColor.g == 0.0f && baseColor.b == 0.0f) {
                materialParams.metallic = 1.0f;
                baseColor = glm::vec4(color.r, color.g, color.b, baseColor.a);
            }
        }
    }
    materialParams.baseColor = baseColor;

    if (aiMat->Get(AI_MATKEY_GLTF_ALPHAMODE, alphaMode) == AI_SUCCESS) {
        if (strcmp(alphaMode.C_Str(), "BLEND") == 0) {
            materialParams.alphaMode = AlphaMode::TRANSPARENT;
        }
        else if (strcmp(alphaMode.C_Str(), "MASK") == 0) {
            materialParams.alphaMode = AlphaMode::MASKED;
            float maskThreshold = 0.5;
            aiMat->Get(AI_MATKEY_GLTF_ALPHACUTOFF, maskThreshold);
            materialParams.maskThreshold = maskThreshold;
        }
    }

    // load base color texture
    if (aiMat->GetTexture(aiTextureType_DIFFUSE, 0, &baseColorPath,
                          nullptr, nullptr, nullptr, nullptr, mapMode) == AI_SUCCESS) {
        TextureID baseColorMap = loadMaterialTexture(aiMat, baseColorPath, true);
        materialParams.albedoTextureID = baseColorMap;
    }

    // load normal map
    if (aiMat->GetTexture(aiTextureType_NORMALS, 0, &normalPath, nullptr,
                          nullptr, nullptr, nullptr, mapMode) == AI_SUCCESS) {
        TextureID normalMap = loadMaterialTexture(aiMat, normalPath);
        materialParams.normalTextureID = normalMap;
    }

    // load metallic and roughness textures
    if (aiMat->GetTexture(aiTextureType_METALNESS, 0, &MPath, nullptr,
                        nullptr, nullptr, nullptr, mapMode) == AI_SUCCESS) {
        TextureID metallicMap = loadMaterialTexture(aiMat, MPath);
        materialParams.metallicTextureID = metallicMap;
    }
    if (aiMat->GetTexture(aiTextureType_DIFFUSE_ROUGHNESS, 0, &RPath, nullptr,
                        nullptr, nullptr, nullptr, mapMode) == AI_SUCCESS) {
        TextureID roughnessMap = loadMaterialTexture(aiMat, RPath);
        materialParams.roughnessTextureID = roughnessMap;
    }
    // if is GLTF and metallic and roughness textures are not set, try to load combined metallic-roughness texture
    if (isGLTF &&
        (materialParams.metallicTextureID == 0 || materialParams.roughnessTextureID == 0)) {
        if (aiMat->GetTexture(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_METALLICROUGHNESS_TEXTURE, &MRPath, nullptr,
                            nullptr, nullptr, nullptr, mapMode) == AI_SUCCESS) {
            TextureID metallicRoughnessMap = loadMaterialTexture(aiMat, MRPath);
            materialParams.metallicTextureID = metallicRoughnessMap;
            materialParams.metalRoughnessCombined = true;
        }
    }

    // load ambient occlusion map
    if (aiMat->GetTexture(aiTextureType_AMBIENT_OCCLUSION, 0, &AOPath, nullptr,
                          nullptr, nullptr, nullptr, mapMode) == AI_SUCCESS) {
        TextureID aoMap = loadMaterialTexture(aiMat, AOPath);
        materialParams.aoTextureID = aoMap;
    }

    // load emissive map
    if (aiMat->GetTexture(aiTextureType_EMISSIVE, 0, &emissivePath, nullptr,
                          nullptr, nullptr, nullptr, mapMode) == AI_SUCCESS) {
        TextureID emissiveMap = loadMaterialTexture(aiMat, emissivePath);
        materialParams.emissiveTextureID = emissiveMap;
    }

    // load factors
    if (aiMat->Get(AI_MATKEY_BASE_COLOR, baseColorFactor) == AI_SUCCESS) {
        materialParams.baseColorFactor = glm::vec4(baseColorFactor.r, baseColorFactor.g, baseColorFactor.b, baseColorFactor.a);
    }

    if (aiMat->Get(AI_MATKEY_COLOR_EMISSIVE, emissiveFactor) == AI_SUCCESS) {
        materialParams.emissiveFactor = glm::vec3(emissiveFactor.r, emissiveFactor.g, emissiveFactor.b);
    }

    if (aiMat->Get(AI_MATKEY_METALLIC_FACTOR, metallicFactor) == AI_SUCCESS) {
        materialParams.metallicFactor = metallicFactor;
    }

    if (aiMat->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughnessFactor) == AI_SUCCESS) {
        materialParams.roughnessFactor = roughnessFactor;
    }
}

int32_t Model::getEmbeddedTextureId(const aiString &path) {
    const char* pathStr = path.C_Str();
    if (path.length >= 2 && pathStr[0] == '*') { // seems like assimp uses * as a prefix for embedded textures
        for (int i = 1; i < path.length; i++) {
            if (!isdigit(pathStr[i])) {
                return -1;
            }
        }
        return std::atoi(pathStr + 1); // NOLINT
    }
    return -1;
}

TextureID Model::loadMaterialTexture(aiMaterial const* aiMat, aiString aiTexturePath, bool shouldGammaCorrect) {
    std::string texturePath = rootDirectory;
    texturePath = texturePath.append(aiTexturePath.C_Str());
    std::replace(texturePath.begin(), texturePath.end(), '\\', '/');

    // if we've loaded this texture already, return the already loaded texture
    if (texturesLoaded.count(texturePath) > 0) {
        return texturesLoaded[texturePath].ID;
    }

    shouldGammaCorrect &= gammaCorrected;

    // if texture is embedded into the file, read it from memory
    int32_t embeddedId = getEmbeddedTextureId(aiTexturePath);
    if (embeddedId != -1) {
        const aiTexture* aiEmbeddedTexture = scene->mTextures[embeddedId];

        int texWidth, texHeight, texChannels;
        unsigned char* data = stbi_load_from_memory(reinterpret_cast<unsigned char*>(aiEmbeddedTexture->pcData),
                                                    aiEmbeddedTexture->mWidth, &texWidth, &texHeight, &texChannels, 0);
        if (data) {
            GLint internalFormat;
            GLenum format;
            if (texChannels == 1) {
                internalFormat = GL_RED;
                format = GL_RED;
            }
            else if (texChannels == 3) {
                internalFormat = shouldGammaCorrect ? GL_SRGB : GL_RGB;
                format = GL_RGB;
            }
            else if (texChannels == 4) {
                internalFormat = shouldGammaCorrect ? GL_SRGB_ALPHA : GL_RGBA;
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

        return 0;
    }
    // else load the texture from external file
    else {
        Texture texture = Texture({
            .wrapS = GL_REPEAT,
            .wrapT = GL_REPEAT,
            .minFilter = GL_LINEAR_MIPMAP_LINEAR,
            .magFilter = GL_LINEAR,
            .gammaCorrected = shouldGammaCorrect,
            .path = texturePath
        });
        texturesLoaded[texturePath] = texture;
        return texturesLoaded[texturePath].ID;
    }
}

Model::~Model() {
    for (auto mesh : meshes) {
        delete mesh;
    }

    for (auto& texture : texturesLoaded) {
        glDeleteTextures(1, &texture.second.ID);
    }
}
