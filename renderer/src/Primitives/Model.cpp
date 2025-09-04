// Adpated from: https://github.com/google/filament/blob/main/libs/filamentapp/src/MeshAssimp.cpp
#include <iostream>
#include <unistd.h>

#include <spdlog/spdlog.h>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/cimport.h>
#include <assimp/GltfMaterial.h>

#include <Utils/FileIO.h>
#include <Primitives/Model.h>

#ifdef __ANDROID__
#include <assimp/port/AndroidJNI/AndroidJNIIOSystem.h>
#endif

using namespace quasar;

Model::Model(const ModelCreateParams& params)
    : flipTextures(params.flipTextures)
    , gammaCorrected(params.gammaCorrected)
    , IBL(params.IBL)
{
    loadFromFile(params);
}

Model::~Model() {
    for (auto mesh : meshes) {
        delete mesh;
    }

    for (auto& texture : texturesLoaded) {
        delete texture.second;
    }
}

Node* Model::findNodeByName(const std::string& name) {
    return rootNode.findNodeByName(name);
}

void Model::updateAnimations(float dt) {
    rootNode.updateAnimations(dt);
}

void Model::loadFromFile(const ModelCreateParams& params) {
    std::string path = params.path;
    spdlog::info("Loading model: {}", path);

    // Use absolute path if path starts with ~/
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
#ifdef __ANDROID__
    Assimp::AndroidJNIIOSystem *ioSystem = new Assimp::AndroidJNIIOSystem(FileIO::getNativeActivity());
    if (ioSystem != nullptr) {
        importer.SetIOHandler(ioSystem);
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

    meshes.resize(scene->mNumMeshes);

    processNode(scene->mRootNode, scene, &rootNode, params.material);

    processAnimations(scene);
}

void Model::processAnimations(const aiScene* scene) {
    for (uint i = 0; i < scene->mNumAnimations; i++) {
        aiAnimation* animation = scene->mAnimations[i];

        for (uint j = 0; j < animation->mNumChannels; ++j) {
            aiNodeAnim* channel = animation->mChannels[j];

            Node* node = rootNode.findNodeByName(channel->mNodeName.C_Str());
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
        node->pushMeshIndex(meshIndex);
    }

    for (int i = 0; i < aiNode->mNumChildren; i++) {
        node->children.push_back(new Node());
        processNode(aiNode->mChildren[i], scene, node->children.back(), material);
    }
}

Mesh* Model::processMesh(aiMesh* mesh, const aiScene* scene, const LitMaterial* material) {
    std::vector<Vertex> vertices;
    std::vector<uint> indices;

    std::vector<aiVector3D> normals(mesh->mNumVertices, aiVector3D(0, 0, 0));

    // Set up indices and manually calculate normals
    for (int i = 0; i < mesh->mNumFaces; i++) {
        const aiFace& face = mesh->mFaces[i];
        const aiVector3D& v0 = mesh->mVertices[face.mIndices[0]];
        const aiVector3D& v1 = mesh->mVertices[face.mIndices[1]];
        const aiVector3D& v2 = mesh->mVertices[face.mIndices[2]];

        aiVector3D normal = (v1 - v0) ^ (v2 - v0);
        normal.Normalize();

        for (int j = 0; j < face.mNumIndices; j++) {
            indices.push_back(face.mIndices[j]);
            normals[face.mIndices[j]] += normal;
        }
    }

    // Set up vertices
    glm::vec3 min = glm::vec3(FLT_MAX);
    glm::vec3 max = glm::vec3(-FLT_MAX);
    for (int i = 0; i < mesh->mNumVertices; i++) {
        Vertex vertex;
        glm::vec3 tmpVec3;

        if (mesh->HasPositions()) {
            tmpVec3.x = mesh->mVertices[i].x;
            tmpVec3.y = mesh->mVertices[i].y;
            tmpVec3.z = mesh->mVertices[i].z;
            vertex.position = tmpVec3;

            min = glm::min(min, tmpVec3);
            max = glm::max(max, tmpVec3);
        }

        if (mesh->HasNormals()) {
            tmpVec3.x = normals[i].x;
            tmpVec3.y = normals[i].y;
            tmpVec3.z = normals[i].z;
            vertex.normal = tmpVec3;
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
            vertex.texCoord = vec;
        }

        if (mesh->HasTangentsAndBitangents()) {
            tmpVec3.x = mesh->mTangents[i].x;
            tmpVec3.y = mesh->mTangents[i].y;
            tmpVec3.z = mesh->mTangents[i].z;
            vertex.tangent = tmpVec3;

            tmpVec3.x = mesh->mBitangents[i].x;
            tmpVec3.y = mesh->mBitangents[i].y;
            tmpVec3.z = mesh->mBitangents[i].z;
            vertex.bitangent = tmpVec3;
        }
        else {
            vertex.bitangent = glm::normalize(glm::cross(vertex.normal, glm::vec3(1.0f, 0.0f, 0.0f)));
            vertex.tangent = glm::normalize(glm::cross(vertex.normal, vertex.bitangent));
        }

        vertices.push_back(vertex);
    }

    // Set up AABB
    aabb.update(min, max);

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

    return new Mesh(meshParams);
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

Texture* Model::loadMaterialTexture(aiMaterial const* aiMat, aiString aiTexturePath, bool shouldGammaCorrect) {
    std::string texturePath = rootDirectory;
    texturePath = texturePath.append(aiTexturePath.C_Str());
    std::replace(texturePath.begin(), texturePath.end(), '\\', '/');

    // If we've loaded this texture already, return the already loaded texture
    if (texturesLoaded.count(texturePath) > 0) {
        return texturesLoaded[texturePath];
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
            texturesLoaded[texturePath] = texture;
            return texturesLoaded[texturePath];
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
        texturesLoaded[texturePath] = texture;
        return texturesLoaded[texturePath];
    }
}

void Model::bindMaterial(Scene& scene, const glm::mat4& model, const Material* overrideMaterial, const Texture* prevIDMap) {
    for (auto& mesh : meshes) {
        mesh->bindMaterial(scene, model, overrideMaterial, prevIDMap);
    }
}

RenderStats Model::draw(GLenum primitiveType, const Camera& camera, const glm::mat4& model, bool frustumCull, const Material* overrideMaterial) {
    RenderStats stats = drawNode(&rootNode, primitiveType, camera, glm::mat4(1.0f), model, frustumCull, overrideMaterial);
    return stats;
}

RenderStats Model::draw(GLenum primitiveType, const Camera& camera, const glm::mat4& model, const BoundingSphere& boundingSphere, const Material* overrideMaterial) {
    RenderStats stats = drawNode(&rootNode, primitiveType, camera, glm::mat4(1.0f), model, boundingSphere, overrideMaterial);
    return stats;
}

RenderStats Model::drawNode(const Node* node,
                            GLenum primitiveType, const Camera& camera,
                            const glm::mat4& parentTransform, const glm::mat4& model,
                            bool frustumCull, const Material* overrideMaterial) {
    RenderStats stats;
    const glm::mat4& globalTransform = parentTransform * node->getTransformParentFromLocal() * node->getTransformAnimation();
    const glm::mat4& modelMatrix = model * globalTransform;

    for (int meshIndex : node->getMeshIndices()) {
        stats += meshes[meshIndex]->draw(primitiveType, camera, modelMatrix, frustumCull, overrideMaterial);
    }

    for (const auto* child : node->children) {
        stats += drawNode(child, primitiveType, camera, globalTransform, model, frustumCull, overrideMaterial);
    }

    return stats;
}

RenderStats Model::drawNode(const Node* node,
                            GLenum primitiveType, const Camera& camera,
                            const glm::mat4& parentTransform, const glm::mat4& model,
                            const BoundingSphere& boundingSphere, const Material* overrideMaterial) {
    RenderStats stats;
    const glm::mat4& globalTransform = parentTransform * node->getTransformParentFromLocal() * node->getTransformAnimation();
    const glm::mat4& modelMatrix = model * globalTransform;

    for (int meshIndex : node->getMeshIndices()) {
        stats += meshes[meshIndex]->draw(primitiveType, camera, modelMatrix, boundingSphere, overrideMaterial);
    }

    for (const auto* child : node->children) {
        stats += drawNode(child, primitiveType, camera, globalTransform, model, boundingSphere, overrideMaterial);
    }

    return stats;
}
