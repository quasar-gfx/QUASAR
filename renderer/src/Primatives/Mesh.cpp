#include <Primatives/Mesh.h>

#include <glm/gtx/string_cast.hpp>

void Mesh::createArrayBuffer() {
    glGenVertexArrays(1, &vertexArrayBuffer);
    createAttributes();
}

void Mesh::createAttributes() {
    glBindVertexArray(vertexArrayBuffer);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);

    glEnableVertexAttribArray(ATTRIBUTE_ID);
    glVertexAttribPointer(ATTRIBUTE_ID,         1, GL_UNSIGNED_INT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, ID));

    glEnableVertexAttribArray(ATTRIBUTE_POSITION);
    glVertexAttribPointer(ATTRIBUTE_POSITION,   3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, position));

    glEnableVertexAttribArray(ATTRIBUTE_COLOR);
    glVertexAttribPointer(ATTRIBUTE_COLOR,      3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, color));

    glEnableVertexAttribArray(ATTRIBUTE_NORMAL);
    glVertexAttribPointer(ATTRIBUTE_NORMAL,     3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));

    glEnableVertexAttribArray(ATTRIBUTE_TEX_COORDS);
    glVertexAttribPointer(ATTRIBUTE_TEX_COORDS, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, texCoords));

    glEnableVertexAttribArray(ATTRIBUTE_TANGENT);
    glVertexAttribPointer(ATTRIBUTE_TANGENT,    3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, tangent));

    glEnableVertexAttribArray(ATTRIBUTE_BITANGENT);
    glVertexAttribPointer(ATTRIBUTE_BITANGENT,  3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, bitangent));

    glBindVertexArray(0);
}

void Mesh::setBuffers(const std::vector<Vertex> &vertices) {
    // if no vertices, return
    if (vertices.size() == 0) {
        return;
    }

    glBindVertexArray(vertexArrayBuffer);

    vertexBuffer.bind();
    vertexBuffer.setData(vertices);
    vertexBuffer.unbind();

    glBindVertexArray(0);

    updateAABB(vertices);
}

void Mesh::setBuffers(const std::vector<Vertex> &vertices, const std::vector<unsigned int> &indices) {
    // if no vertices or indices, dont bind buffers
    if (vertices.size() == 0) {
        return;
    }

    glBindVertexArray(vertexArrayBuffer);

    vertexBuffer.bind();
    vertexBuffer.setData(vertices);
    vertexBuffer.unbind();

    updateAABB(vertices);

    if (indices.size() == 0) {
        glBindVertexArray(0);
        return;
    }

    indexBuffer.bind();
    indexBuffer.setData(indices);
    indexBuffer.unbind();

    glBindVertexArray(0);
}

void Mesh::resizeBuffers(unsigned int vertexBufferSize, unsigned int indexBufferSize) {
    vertexBuffer.setSize(vertexBufferSize);
    indexBuffer.setSize(indexBufferSize);
}

void Mesh::updateAABB(const std::vector<Vertex> &vertices) {
    glm::vec3 min = vertices[0].position;
    glm::vec3 max = vertices[0].position;

    for (auto& vertex : vertices) {
        min = glm::min(min, vertex.position);
        max = glm::max(max, vertex.position);
    }

    // set up AABB
    aabb.update(min, max);
}

void Mesh::setMaterialCameraParams(const Camera &camera, const Material* material) {
    if (camera.isVR()) {
        auto& vrCamera = static_cast<const VRCamera&>(camera);
        material->shader->setMat4("view[0]", vrCamera.left.getViewMatrix());
        material->shader->setMat4("projection[0]", vrCamera.left.getProjectionMatrix());
        material->shader->setMat4("view[1]", vrCamera.right.getViewMatrix());
        material->shader->setMat4("projection[1]", vrCamera.right.getProjectionMatrix());
        material->shader->setVec3("camPos", vrCamera.getPosition());
    }
    else {
        auto& monoCamera = static_cast<const PerspectiveCamera&>(camera);
        material->shader->setMat4("view", monoCamera.getViewMatrix());
        material->shader->setMat4("projection", monoCamera.getProjectionMatrix());
        material->shader->setVec3("camPos", monoCamera.getPosition());
    }
}

void Mesh::bindMaterial(const Scene &scene, const glm::mat4 &model, const Material* overrideMaterial, const Texture* prevDepthMap) {
    auto materialToUse = overrideMaterial != nullptr ? overrideMaterial : material;
    materialToUse->bind();

    if (scene.ambientLight != nullptr) {
        scene.ambientLight->bindMaterial(materialToUse);
    }

    scene.bindMaterial(materialToUse);

    int texIdx = materialToUse->getTextureCount() + Scene::numTextures;
    if (scene.directionalLight != nullptr) {
        scene.directionalLight->bindMaterial(materialToUse);
        if (overrideMaterial != nullptr) {
            materialToUse->shader->setMat4("lightSpaceMatrix", scene.directionalLight->lightSpaceMatrix * model);
        }
        else {
            materialToUse->shader->setTexture("dirLightShadowMap", scene.directionalLight->shadowMapRenderTarget.depthBuffer, texIdx);
            materialToUse->shader->setMat4("lightSpaceMatrix", scene.directionalLight->lightSpaceMatrix);
        }
    }
    texIdx++;

    for (int i = 0; i < scene.pointLights.size(); i++) {
        auto pointLight = scene.pointLights[i];
        pointLight->setChannel(i);
        materialToUse->shader->setTexture("pointLightShadowMaps[" + std::to_string(i) + "]", pointLight->shadowMapRenderTarget.depthCubeMap, texIdx);
        pointLight->bindMaterial(materialToUse);
        texIdx++;
    }

    materialToUse->shader->setInt("numPointLights", static_cast<int>(scene.pointLights.size()));
    materialToUse->shader->setFloat("material.IBL", IBL);

    materialToUse->shader->setFloat("pointSize", pointSize);

    materialToUse->shader->setBool("peelDepth", prevDepthMap != nullptr);
    if (prevDepthMap != nullptr) {
        materialToUse->shader->setTexture("prevDepthMap", *prevDepthMap, texIdx);
        texIdx++;
    }

    materialToUse->unbind();
}

RenderStats Mesh::draw(const Camera &camera, const glm::mat4 &model, bool frustumCull, const Material* overrideMaterial) {
    RenderStats stats;

    if (camera.isVR()) {
        auto vrcamera = static_cast<const VRCamera*>(&camera);
        auto &frustumLeft = vrcamera->left.frustum;
        auto &frustumRight = vrcamera->right.frustum;
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

    setMaterialCameraParams(camera, materialToUse);
    materialToUse->shader->setMat4("model", model);
    materialToUse->shader->setMat3("normalMatrix", glm::transpose(glm::inverse(glm::mat3(model))));

    stats = draw();

    materialToUse->unbind();

    return stats;
}

RenderStats Mesh::draw(const Camera &camera, const glm::mat4 &model, const BoundingSphere &boundingSphere, const Material* overrideMaterial) {
    RenderStats stats;
    if (!boundingSphere.intersects(aabb)) {
        return stats;
    }
    return draw(camera, model, false, overrideMaterial);
}

RenderStats Mesh::draw() {
    GLenum primativeType = pointcloud ? GL_POINTS : GL_TRIANGLES;

    glBindVertexArray(vertexArrayBuffer);
    if (indexBuffer.getSize() > 0) {
        indexBuffer.bind();
        glDrawElements(primativeType, indexBuffer.getSize(), GL_UNSIGNED_INT, 0);
        indexBuffer.unbind();
    }
    else {
        vertexBuffer.bind();
        glDrawArrays(primativeType, 0, vertexBuffer.getSize());
        vertexBuffer.unbind();
    }
    glBindVertexArray(0);

    RenderStats stats;
    if (indexBuffer.getSize() > 0) {
        stats.trianglesDrawn = static_cast<unsigned int>(indexBuffer.getSize() / 3);
    }
    else {
        stats.trianglesDrawn = static_cast<unsigned int>(vertexBuffer.getSize() / 3);
    }
    stats.drawCalls = 1;

    return stats;
}
