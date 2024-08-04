#include <Primatives/Mesh.h>

#include <glm/gtx/string_cast.hpp>

uint32_t Vertex::nextID = 0;

void Mesh::createBuffers()  {
    glGenVertexArrays(1, &vertexArrayBuffer);
    glGenBuffers(1, &vertexBuffer);
    glGenBuffers(1, &indexBuffer);

    updateBuffers();
    createAttributes();
}

void Mesh::createAttributes() {
    glBindVertexArray(vertexArrayBuffer);

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
    glVertexAttribPointer(ATTRIBUTE_BITANGENT,  4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, bitangent));

    glBindVertexArray(0);
}

void Mesh::updateBuffers() {
    glBindVertexArray(vertexArrayBuffer);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    glBindVertexArray(0);
}

void Mesh::setBuffers(const std::vector<Vertex> &vertices, const std::vector<unsigned int> &indices) {
    this->vertices = vertices;
    this->indices = indices;

    updateBuffers();
    updateAABB();
}

void Mesh::setBuffers(GLuint vertexBufferSSBO, GLuint indexBufferSSBO) {
    // copy vertex buffer
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, vertexBufferSSBO);
    glBindBuffer(GL_COPY_READ_BUFFER, vertexBufferSSBO);
    glBindBuffer(GL_COPY_WRITE_BUFFER, vertexBuffer);
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, vertices.size() * sizeof(Vertex));

    if (indexBufferSSBO == -1) {
        return;
    }

    // copy index buffer
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, indexBufferSSBO);
    glBindBuffer(GL_COPY_READ_BUFFER, indexBufferSSBO);
    glBindBuffer(GL_COPY_WRITE_BUFFER, indexBuffer);
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, indices.size() * sizeof(unsigned int));
}

void Mesh::updateAABB() {
    if (vertices.empty()) {
        return;
    }

    glm::vec3 min = vertices[0].position;
    glm::vec3 max = vertices[0].position;

    for (auto& vertex : vertices) {
        min = glm::min(min, vertex.position);
        max = glm::max(max, vertex.position);
    }

    // set up AABB
    aabb.update(min, max);
}

void Mesh::bindMaterial(const Scene &scene, const Camera &camera, const glm::mat4 &model, const Material* overrideMaterial) {
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

    materialToUse->unbind();
}

RenderStats Mesh::draw(const Scene &scene, const Camera &camera, const glm::mat4 &model, bool frustumCull, const Material* overrideMaterial) {
    RenderStats stats;

    auto& frustum = camera.frustum;
    if (frustumCull && !frustum.aabbIsVisible(aabb, model)) {
        return stats;
    }

    auto materialToUse = overrideMaterial != nullptr ? overrideMaterial : material;
    materialToUse->bind();

    materialToUse->shader->setMat4("view", camera.getViewMatrix());
    materialToUse->shader->setMat4("projection", camera.getProjectionMatrix());
    materialToUse->shader->setVec3("camPos", camera.getPosition());
    materialToUse->shader->setMat4("model", model);
    materialToUse->shader->setMat3("normalMatrix", glm::transpose(glm::inverse(glm::mat3(model))));

#ifndef __ANDROID__
    if (wireframe) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    }
#endif

    GLenum primativeType = pointcloud ? GL_POINTS : GL_TRIANGLES;

    glBindVertexArray(vertexArrayBuffer);
    if (indices.size() > 0) {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
        glDrawElements(primativeType, static_cast<unsigned int>(indices.size()), GL_UNSIGNED_INT, 0);
    }
    else {
        glDrawArrays(primativeType, 0, static_cast<unsigned int>(vertices.size()));
    }
    glBindVertexArray(0);

#ifndef __ANDROID__
    if (wireframe) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
#endif

    materialToUse->unbind();

    if (indices.size() > 0) {
        stats.trianglesDrawn = static_cast<unsigned int>(indices.size() / 3);
    }
    else {
        stats.trianglesDrawn = static_cast<unsigned int>(vertices.size() / 3);
    }
    stats.drawCalls = 1;

    return stats;
}

RenderStats Mesh::draw(const Scene &scene, const Camera &camera, const glm::mat4 &model, const BoundingSphere &boundingSphere, const Material* overrideMaterial) {
    RenderStats stats;
    if (!boundingSphere.intersects(aabb)) {
        return stats;
    }
    return draw(scene, camera, model, false, overrideMaterial);
}
