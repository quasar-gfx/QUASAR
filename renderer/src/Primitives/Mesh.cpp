#include <Primitives/Mesh.h>

#include <glm/gtx/string_cast.hpp>

void Mesh::setArrayBufferAttributes(const VertexInputAttributes &attributes, unsigned int vertexSize) {
    glGenVertexArrays(1, &vertexArrayBuffer);
    glBindVertexArray(vertexArrayBuffer);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);

    if (attributes.size() == 0) {
        spdlog::warn("No vertex attributes provided!");
    }
    for (auto& attribute : attributes) {
        glEnableVertexAttribArray(attribute.index);
        glVertexAttribPointer(attribute.index, attribute.size, attribute.type, attribute.normalized, vertexSize, (void*)attribute.pointer);
    }

    glBindVertexArray(0);
}

void Mesh::setBuffers(const void* verticesData, unsigned int verticesSize, const unsigned int* indicesData, unsigned int indicesSize) {
    // if no vertices, dont bind buffers
    if (verticesData == nullptr || verticesSize == 0) {
        return;
    }

    glBindVertexArray(vertexArrayBuffer);

    vertexBuffer.bind();
    vertexBuffer.setData(verticesSize, verticesData);

    updateAABB(verticesData, verticesSize);

    if (indicesData == nullptr || indicesSize == 0) {
        glBindVertexArray(0);
        return;
    }

    indexBuffer.bind();
    indexBuffer.setData(indicesSize, indicesData);

    glBindVertexArray(0);
}

void Mesh::setBuffers(unsigned int verticesSize, unsigned int indicesSize) {
    // if no vertices or indices, dont bind buffers
    if (verticesSize == 0) {
        return;
    }

    glBindVertexArray(vertexArrayBuffer);

    vertexBuffer.bind();
    vertexBuffer.resize(verticesSize);

    if (indicesSize == 0) {
        glBindVertexArray(0);
        return;
    }

    indexBuffer.bind();
    indexBuffer.resize(indicesSize);

    glBindVertexArray(0);
}

void Mesh::resizeBuffers(unsigned int verticesSize, unsigned int indicesSize) {
    vertexBuffer.resize(verticesSize);
    indexBuffer.resize(indicesSize);
}

void Mesh::updateAABB(const void* verticesData, unsigned int verticesSize) {
    // if no vertices, return
    if (verticesData == nullptr || verticesSize == 0) {
        return;
    }

    auto* verticesVec = reinterpret_cast<const Vertex*>(verticesData);
    glm::vec3 min = verticesVec[0].position;
    glm::vec3 max = verticesVec[0].position;

    for (unsigned int i = 1; i < verticesSize; i++) {
        auto &vertex = verticesVec[i];
        min = glm::min(min, vertex.position);
        max = glm::max(max, vertex.position);
    }

    // set up AABB
    aabb.update(min, max);
}

void Mesh::setMaterialCameraParams(const Camera &camera, const Material* material) {
    if (camera.isVR()) {
        auto& vrCamera = static_cast<const VRCamera&>(camera);
        material->getShader()->setMat4("camera.view[0]", vrCamera.left.getViewMatrix());
        material->getShader()->setMat4("camera.projection[0]", vrCamera.left.getProjectionMatrix());
        material->getShader()->setMat4("camera.view[1]", vrCamera.right.getViewMatrix());
        material->getShader()->setMat4("camera.projection[1]", vrCamera.right.getProjectionMatrix());
    }
    else {
        auto& monoCamera = static_cast<const PerspectiveCamera&>(camera);
        material->getShader()->setMat4("camera.view", monoCamera.getViewMatrix());
        material->getShader()->setMat4("camera.projection", monoCamera.getProjectionMatrix());
    }
    material->getShader()->setVec3("camera.position", camera.getPosition());
    material->getShader()->setFloat("camera.fovy", camera.getFovyRadians());
    material->getShader()->setFloat("camera.near", camera.getNear());
    material->getShader()->setFloat("camera.far", camera.getFar());
}

void Mesh::bindMaterial(const Scene &scene, const glm::mat4 &model, const Material* overrideMaterial, const Texture* prevIDMap) {
    auto* materialToUse = overrideMaterial != nullptr ? overrideMaterial : material;
    materialToUse->bind();

    scene.bindMaterial(materialToUse);

    if (scene.ambientLight != nullptr) {
        scene.ambientLight->bindMaterial(materialToUse);
    }

    int texIdx = materialToUse->getTextureCount() + Scene::numTextures;
    if (scene.directionalLight != nullptr) {
        scene.directionalLight->bindMaterial(materialToUse);
        materialToUse->getShader()->setMat4("lightSpaceMatrix", scene.directionalLight->lightSpaceMatrix);
        if (overrideMaterial == nullptr) {
            materialToUse->getShader()->setTexture("dirLightShadowMap", scene.directionalLight->shadowMapRenderTarget.depthBuffer, texIdx);
        }
    }
    texIdx++;

    for (int i = 0; i < scene.pointLights.size(); i++) {
        auto pointLight = scene.pointLights[i];
        pointLight->setChannel(i);
        materialToUse->getShader()->setTexture("pointLightShadowMaps[" + std::to_string(i) + "]", pointLight->shadowMapRenderTarget.depthCubeMap, texIdx);
        pointLight->bindMaterial(materialToUse);
        texIdx++;
    }

    materialToUse->getShader()->setInt("numPointLights", static_cast<int>(scene.pointLights.size()));
    materialToUse->getShader()->setFloat("material.IBL", IBL);

    materialToUse->getShader()->setBool("peelDepth", prevIDMap != nullptr);
    if (prevIDMap != nullptr) {
        materialToUse->getShader()->setTexture("prevIDMap", *prevIDMap, texIdx);
        texIdx++;
    }

    materialToUse->unbind();
}

RenderStats Mesh::draw(GLenum primativeType, const Camera &camera, const glm::mat4 &model, bool frustumCull, const Material* overrideMaterial) {
    RenderStats stats;

    if (camera.isVR()) {
        auto vrcamera = static_cast<const VRCamera*>(&camera);
        auto& frustumLeft = vrcamera->left.frustum;
        auto& frustumRight = vrcamera->right.frustum;
        if (frustumCull && !frustumLeft.aabbIsVisible(aabb, model) && !frustumRight.aabbIsVisible(aabb, model)) {
            return stats;
        }
    }
    else {
        auto monocamera = static_cast<const PerspectiveCamera*>(&camera);
        auto& frustum = monocamera->frustum;
        if (frustumCull && !frustum.aabbIsVisible(aabb, model)) {
            return stats;
        }
    }

    auto materialToUse = overrideMaterial != nullptr ? overrideMaterial : material;
    materialToUse->bind();

    // set draw ID/object ID
    materialToUse->getShader()->setUint("drawID", ID);

    // set camera params
    setMaterialCameraParams(camera, materialToUse);

    // set model and normal matrix
    materialToUse->getShader()->setMat4("model", model);
    materialToUse->getShader()->setMat3("normalMatrix", glm::transpose(glm::inverse(glm::mat3(model))));

    stats = draw(primativeType);

    materialToUse->unbind();

    return stats;
}

RenderStats Mesh::draw(GLenum primativeType, const Camera &camera, const glm::mat4 &model, const BoundingSphere &boundingSphere, const Material* overrideMaterial) {
    RenderStats stats;
    if (!boundingSphere.intersects(model, aabb)) {
        return stats;
    }
    return draw(primativeType, camera, model, false, overrideMaterial);
}

RenderStats Mesh::draw(GLenum primativeType) {
    RenderStats stats;

    glBindVertexArray(vertexArrayBuffer);
    if (indirectDraw) {
        indirectBuffer.bind();
        if (indexBuffer.getSize() > 0) {
            indexBuffer.bind();
            glDrawElementsIndirect(primativeType, GL_UNSIGNED_INT, 0);
        }
        else {
            vertexBuffer.bind();
            glDrawArraysIndirect(primativeType, 0);
        }
    }
    else {
        if (indexBuffer.getSize() > 0) {
            indexBuffer.bind();
            glDrawElements(primativeType, indexBuffer.getSize(), GL_UNSIGNED_INT, 0);
        }
        else {
            vertexBuffer.bind();
            glDrawArrays(primativeType, 0, vertexBuffer.getSize());
        }
    }
    glBindVertexArray(0);

    if (indexBuffer.getSize() > 0) {
        stats.trianglesDrawn = static_cast<unsigned int>(indexBuffer.getSize() / 3);
    }
    else {
        stats.trianglesDrawn = static_cast<unsigned int>(vertexBuffer.getSize() / 3);
    }
    stats.drawCalls = 1;

    return stats;
}
