// Adpated from: https://github.com/google/filament/blob/main/libs/filamentapp/src/MeshAssimp.cpp
#include <spdlog/spdlog.h>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/cimport.h>
#include <assimp/GltfMaterial.h>
#ifdef __ANDROID__
#include <assimp/port/AndroidJNI/AndroidJNIIOSystem.h>
#endif

#include <Path.h>
#include <Utils/FileIO.h>
#include <Primitives/Model.h>

using namespace quasar;

Model::Model(const ModelCreateParams& params)
    : flipTextures(params.flipTextures)
    , gammaCorrected(params.gammaCorrected)
    , IBL(params.IBL)
{
    loadFromFile(params);
}

Model::~Model() {
    for (auto* mesh : meshes) {
        delete mesh;
    }
    for (auto& texture : texturesCache) {
        delete texture.second;
    }
}

void Model::loadFromFile(const ModelCreateParams& params) {
    Path path(params.path);
    spdlog::info("Loading model: {}", path.str());

    // Get the absolute path to the model file
    std::string absolutePath = path.absolutePathStr();

    // Create importer
    Assimp::Importer importer;
    importer.SetPropertyInteger(AI_CONFIG_PP_SBP_REMOVE, aiPrimitiveType_LINE | aiPrimitiveType_POINT);
    importer.SetPropertyBool(AI_CONFIG_IMPORT_COLLADA_IGNORE_UP_DIRECTION, true);
    importer.SetPropertyBool(AI_CONFIG_PP_PTV_KEEP_HIERARCHY, true);
#ifdef __ANDROID__
    Assimp::AndroidJNIIOSystem *ioSystem = new Assimp::AndroidJNIIOSystem(FileIO::getNativeActivity());
    if (ioSystem != nullptr) {
        importer.SetIOHandler(ioSystem);
    }

    // Android expects paths relative to assets/ folder so remove starting /
    if (!absolutePath.empty() && absolutePath[0] == '/') {
        absolutePath = absolutePath.substr(1);
    }
#endif

    uint flags = \
            // Normals and tangents
            aiProcess_GenSmoothNormals |
            aiProcess_CalcTangentSpace |
            // UV coordinates
            aiProcess_GenUVCoords |
            // Topology optimization
            aiProcess_FindInstances |
            aiProcess_OptimizeMeshes |
            aiProcess_JoinIdenticalVertices |
            // Misc optimization
            aiProcess_ImproveCacheLocality |
            aiProcess_SortByPType |
            // We only support triangles
            aiProcess_Triangulate;
    scene = importer.ReadFile(absolutePath, flags);
    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        throw std::runtime_error("ERROR::ASSIMP:: " + std::string(importer.GetErrorString()));
    }

    std::string extension = path.extension();
    size_t index = importer.GetImporterIndex(extension.c_str());
    const aiImporterDesc* importerDesc = importer.GetImporterInfo(index);
    isGLTF = importerDesc &&
            (!strncmp("glTF Importer",  importerDesc->mName, 13) ||
             !strncmp("glTF2 Importer", importerDesc->mName, 14));

    rootDirectory = path.parent().str();

    meshes.resize(scene->mNumMeshes);

    processNode(scene->mRootNode, scene, this, params.material);
    processAnimations(scene);
}

void Model::processAnimations(const aiScene* scene) {
    for (uint i = 0; i < scene->mNumAnimations; i++) {
        aiAnimation* animation = scene->mAnimations[i];

        for (uint j = 0; j < animation->mNumChannels; ++j) {
            aiNodeAnim* channel = animation->mChannels[j];

            Node* node = findNodeByName(channel->mNodeName.C_Str());
            if (node == nullptr) {
                spdlog::warn("Node {} not found in model", channel->mNodeName.C_Str());
                continue;
            }

            std::shared_ptr<Animation> anim = node->addAnimation();
            const glm::mat4& transformInv = node->getTransformLocalFromParent();

            for (uint k = 0; k < channel->mNumPositionKeys; k++) {
                aiVectorKey positionKey = channel->mPositionKeys[k];
                const glm::vec3 pos = glm::vec3(positionKey.mValue.x, positionKey.mValue.y, positionKey.mValue.z);
                const glm::vec3 adjustedPos = glm::vec3(transformInv * glm::vec4(pos, 1.0f));
                anim->addPositionKey(adjustedPos, positionKey.mTime / animation->mTicksPerSecond);
            }

            for (uint k = 0; k < channel->mNumRotationKeys; k++) {
                aiQuatKey rotationKey = channel->mRotationKeys[k];
                const glm::quat rot = glm::quat(rotationKey.mValue.w, rotationKey.mValue.x, rotationKey.mValue.y, rotationKey.mValue.z);
                const glm::quat adjustedRot = glm::quat(transformInv) * rot;
                anim->addRotationKey(glm::degrees(glm::eulerAngles(adjustedRot)), rotationKey.mTime / animation->mTicksPerSecond);
            }

            for (uint k = 0; k < channel->mNumScalingKeys; k++) {
                aiVectorKey scalingKey = channel->mScalingKeys[k];
                const glm::vec3 scale = glm::vec3(scalingKey.mValue.x, scalingKey.mValue.y, scalingKey.mValue.z);
                anim->addScaleKey(scale, scalingKey.mTime / animation->mTicksPerSecond);
            }
        }
    }
}

void Model::processNode(aiNode* aiNode, const aiScene* scene, Node* node, const LitMaterial* material) {
    const glm::mat4& transform = glm::transpose(reinterpret_cast<glm::mat4&>(aiNode->mTransformation));

    node->setName(aiNode->mName.C_Str());
    node->setTransformParentFromLocal(transform);

    for (int i = 0; i < aiNode->mNumMeshes; i++) {
        const int meshIndex = aiNode->mMeshes[i];
        aiMesh* mesh = scene->mMeshes[meshIndex];
        meshes[meshIndex] = processMesh(mesh, scene, material);
        node->addEntity(meshes[meshIndex]);
    }

    // Process child nodes
    for (int i = 0; i < aiNode->mNumChildren; i++) {
        Node* childNode = new Node();
        node->addChildNode(childNode);
        processNode(aiNode->mChildren[i], scene, childNode, material);
    }
}

Mesh* Model::processMesh(aiMesh* mesh, const aiScene* scene, const LitMaterial* material) {
    std::vector<Vertex> vertices(mesh->mNumVertices);
    std::vector<uint> indices(mesh->mNumFaces * 3); // Assume triangles

    // Set up indices and manually calculate normals
    std::vector<aiVector3D> normals(mesh->mNumVertices, aiVector3D(0, 0, 0));
    for (int i = 0; i < mesh->mNumFaces; i++) {
        const aiFace& face = mesh->mFaces[i];
        const aiVector3D& v0 = mesh->mVertices[face.mIndices[0]];
        const aiVector3D& v1 = mesh->mVertices[face.mIndices[1]];
        const aiVector3D& v2 = mesh->mVertices[face.mIndices[2]];

        aiVector3D normal = (v1 - v0) ^ (v2 - v0);
        normal.Normalize();

        for (int j = 0; j < face.mNumIndices; j++) {
            indices[i * 3 + j] = face.mIndices[j];
            normals[face.mIndices[j]] += normal;
        }
    }

    // Set up vertices
    glm::vec3 min = glm::vec3(FLT_MAX);
    glm::vec3 max = glm::vec3(-FLT_MAX);
    for (int i = 0; i < mesh->mNumVertices; i++) {
        if (mesh->HasPositions()) {
            vertices[i].position.x = mesh->mVertices[i].x;
            vertices[i].position.y = mesh->mVertices[i].y;
            vertices[i].position.z = mesh->mVertices[i].z;

            min = glm::min(min, vertices[i].position);
            max = glm::max(max, vertices[i].position);
        }

        if (mesh->HasNormals()) {
            vertices[i].normal.x = normals[i].x;
            vertices[i].normal.y = normals[i].y;
            vertices[i].normal.z = normals[i].z;
        }

        if (mesh->HasTextureCoords(0)) {
            vertices[i].texCoord.x = mesh->mTextureCoords[0][i].x;
            if (flipTextures) {
                vertices[i].texCoord.y = 1.0f - mesh->mTextureCoords[0][i].y;
            }
            else {
                vertices[i].texCoord.y = mesh->mTextureCoords[0][i].y;
            }
        }

        if (mesh->HasTangentsAndBitangents()) {
            vertices[i].tangent.x = mesh->mTangents[i].x;
            vertices[i].tangent.y = mesh->mTangents[i].y;
            vertices[i].tangent.z = mesh->mTangents[i].z;

            vertices[i].bitangent.x = mesh->mBitangents[i].x;
            vertices[i].bitangent.y = mesh->mBitangents[i].y;
            vertices[i].bitangent.z = mesh->mBitangents[i].z;
        }
        else {
            vertices[i].bitangent = glm::normalize(glm::cross(vertices[i].normal, glm::vec3(1.0f, 0.0f, 0.0f)));
            vertices[i].tangent   = glm::normalize(glm::cross(vertices[i].normal, vertices[i].bitangent));
        }
    }

    // Set up material
    uint32_t materialId = mesh->mMaterialIndex;
    aiMaterial const* aiMat = scene->mMaterials[materialId];

    MeshDataCreateParams meshParams{};
    if (material != nullptr) {
        this->material = material;
    }
    else {
        LitMaterialCreateParams materialParams{};
        processMaterial(aiMat, materialParams);
        this->material = new LitMaterial(materialParams);
    }

    meshParams.verticesData = vertices.data();
    meshParams.verticesSize = vertices.size();
    meshParams.indicesData = indices.data();
    meshParams.indicesSize = indices.size();
    meshParams.IBL = IBL;
    meshParams.material = this->material;

    auto* result = new Mesh(meshParams);
    result->updateAABB(min, max);
    return result;
}

void Model::processMaterial(const aiMaterial* aiMat, LitMaterialCreateParams& materialParams) {
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

    // Convert shininess to roughness
    float roughness = sqrt(2.0f / (shininess + 2.0f));
    materialParams.roughness = roughness;

    materialParams.metallic = 0.0f;
    if (aiMat->Get(AI_MATKEY_COLOR_SPECULAR, color) == AI_SUCCESS) {
        // If there's a non-grey specular color, assume a metallic surface
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

    // Load base color texture
    if (aiMat->GetTexture(aiTextureType_DIFFUSE, 0, &baseColorPath,
                          nullptr, nullptr, nullptr, nullptr, mapMode) == AI_SUCCESS) {
        Texture* baseColorMap = loadMaterialTexture(aiMat, baseColorPath, true);
        materialParams.albedoTexture = baseColorMap;
    }

    // Load normal map
    if (aiMat->GetTexture(aiTextureType_NORMALS, 0, &normalPath, nullptr,
                          nullptr, nullptr, nullptr, mapMode) == AI_SUCCESS) {
        Texture* normalMap = loadMaterialTexture(aiMat, normalPath);
        materialParams.normalTexture = normalMap;
    }

    // If model is GLTF, try to load combined metallic-roughness texture
    if (isGLTF && aiMat->GetTexture(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_METALLICROUGHNESS_TEXTURE, &MRPath, nullptr,
                                    nullptr, nullptr, nullptr, mapMode) == AI_SUCCESS) {
        Texture* metallicRoughnessMap = loadMaterialTexture(aiMat, MRPath);
        materialParams.metallicTexture = metallicRoughnessMap;
        materialParams.metalRoughnessCombined = true;
    }
    // If not GLTF or there is no combined texture, load metallic and roughness textures separately
    else {
        if (aiMat->GetTexture(aiTextureType_METALNESS, 0, &MPath, nullptr,
                              nullptr, nullptr, nullptr, mapMode) == AI_SUCCESS) {
            Texture* metallicMap = loadMaterialTexture(aiMat, MPath);
            materialParams.metallicTexture = metallicMap;
        }
        if (aiMat->GetTexture(aiTextureType_DIFFUSE_ROUGHNESS, 0, &RPath, nullptr,
                              nullptr, nullptr, nullptr, mapMode) == AI_SUCCESS) {
            Texture* roughnessMap = loadMaterialTexture(aiMat, RPath);
            materialParams.roughnessTexture = roughnessMap;
        }
    }

    // Load ambient occlusion map
    if (aiMat->GetTexture(aiTextureType_LIGHTMAP, 0, &AOPath, nullptr,
                          nullptr, nullptr, nullptr, mapMode) == AI_SUCCESS) {
        Texture* aoMap = loadMaterialTexture(aiMat, AOPath);
        materialParams.aoTexture = aoMap;
    }

    // Load emissive map
    if (aiMat->GetTexture(aiTextureType_EMISSIVE, 0, &emissivePath, nullptr,
                          nullptr, nullptr, nullptr, mapMode) == AI_SUCCESS) {
        Texture* emissiveMap = loadMaterialTexture(aiMat, emissivePath);
        materialParams.emissiveTexture = emissiveMap;
    }

    // Load factors
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

int32_t Model::getEmbeddedTextureId(const aiString& path) {
    const char* pathStr = path.C_Str();
    if (path.length >= 2 && pathStr[0] == '*') { // assimp uses * as a prefix for embedded textures
        for (int i = 1; i < path.length; i++) {
            if (!isdigit(pathStr[i])) {
                return -1;
            }
        }
        return std::atoi(pathStr + 1);
    }
    return -1;
}

Texture* Model::loadMaterialTexture(aiMaterial const* aiMat, aiString aiTexturePath, bool shouldGammaCorrect) {
    std::string texturePath = rootDirectory;
    texturePath = texturePath.append(aiTexturePath.C_Str());
    std::replace(texturePath.begin(), texturePath.end(), '\\', '/');

    // If we've loaded this texture already, return the already loaded texture
    if (texturesCache.count(texturePath) > 0) {
        return texturesCache[texturePath];
    }

    shouldGammaCorrect &= gammaCorrected;

    // If texture is embedded into the file, read it from memory
    int32_t embeddedId = getEmbeddedTextureId(aiTexturePath);
    if (embeddedId != -1) {
        const aiTexture* aiEmbeddedTexture = scene->mTextures[embeddedId];

        int texWidth, texHeight, texChannels;
        unsigned char* data = FileIO::loadImageFromMemory(reinterpret_cast<unsigned char*>(aiEmbeddedTexture->pcData),
                                                          aiEmbeddedTexture->mWidth,
                                                          &texWidth, &texHeight, &texChannels, 0);
        if (data) {
            GLint internalFormat;
            GLenum format;
            if (texChannels == 1) {
                internalFormat = GL_R8;
                format = GL_R8;
            }
            else if (texChannels == 3) {
                internalFormat = shouldGammaCorrect ? GL_SRGB : GL_RGB;
                format = GL_RGB;
            }
            else if (texChannels == 4) {
                internalFormat = shouldGammaCorrect ? GL_SRGB8_ALPHA8 : GL_RGBA;
                format = GL_RGBA;
            }

            Texture* texture = new Texture({
                .width = static_cast<uint>(texWidth),
                .height = static_cast<uint>(texHeight),
                .internalFormat = internalFormat,
                .format = format,
                .wrapS = GL_REPEAT,
                .wrapT = GL_REPEAT,
                .minFilter = GL_LINEAR_MIPMAP_LINEAR,
                .magFilter = GL_LINEAR,
                .alignment = 1,
                .data = data,
            });

            FileIO::freeImage(data);
            texturesCache[texturePath] = texture;
            return texturesCache[texturePath];
        }

        return nullptr;
    }
    // Else load the texture from external file
    else {
        Texture* texture = new Texture({
            .wrapS = GL_REPEAT,
            .wrapT = GL_REPEAT,
            .minFilter = GL_LINEAR_MIPMAP_LINEAR,
            .magFilter = GL_LINEAR,
            .gammaCorrected = shouldGammaCorrect,
            .path = texturePath,
        });
        texturesCache[texturePath] = texture;
        return texturesCache[texturePath];
    }
}
