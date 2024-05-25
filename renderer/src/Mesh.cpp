#include <Primatives/Mesh.h>

#include <glm/gtx/string_cast.hpp>

uint32_t Vertex::nextID = 0;

void Mesh::createBuffers()  {
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(ATTRIBUTE_ID);
    glVertexAttribPointer(ATTRIBUTE_ID,         1, GL_UNSIGNED_INT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, ID));

    glEnableVertexAttribArray(ATTRIBUTE_POSITION);
    glVertexAttribPointer(ATTRIBUTE_POSITION,   3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, position));

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

void Mesh::bindSceneAndCamera(Scene &scene, Camera &camera, glm::mat4 model, Material* overrideMaterial) {
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
            materialToUse->shader->setInt("dirLightShadowMap", texIdx);
            scene.directionalLight->shadowMapRenderTarget.depthBuffer.bind(texIdx);
            materialToUse->shader->setMat4("lightSpaceMatrix", scene.directionalLight->lightSpaceMatrix);
        }
    }
    texIdx++;

    for (int i = 0; i < scene.pointLights.size(); i++) {
        auto pointLight = scene.pointLights[i];
        pointLight->setChannel(i);
        materialToUse->shader->setInt("pointLightShadowMaps[" + std::to_string(i) + "]", texIdx);
        pointLight->shadowMapRenderTarget.depthCubeMap.bind(texIdx);
        pointLight->bindMaterial(materialToUse);
        texIdx++;
    }

    materialToUse->shader->setMat4("view", camera.getViewMatrix());
    materialToUse->shader->setMat4("projection", camera.getProjectionMatrix());
    materialToUse->shader->setVec3("camPos", camera.position);
    materialToUse->shader->setMat4("model", model);
    materialToUse->shader->setMat3("normalMatrix", glm::transpose(glm::inverse(glm::mat3(model))));

    materialToUse->shader->setFloat("IBL", IBL);

    materialToUse->shader->setInt("numPointLights", static_cast<int>(scene.pointLights.size()));

    materialToUse->unbind();
}

void Mesh::draw(Material* overrideMaterial) {
    auto materialToUse = overrideMaterial != nullptr ? overrideMaterial : material;
    materialToUse->bind();

    if (wireframe) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    }

    GLenum primativeType = pointcloud ? GL_POINTS : GL_TRIANGLES;

    glBindVertexArray(VAO);
    if (indices.size() > 0) {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glDrawElements(primativeType, static_cast<unsigned int>(indices.size()), GL_UNSIGNED_INT, 0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    }
    else {
        glDrawArrays(primativeType, 0, static_cast<unsigned int>(vertices.size()));
    }
    glBindVertexArray(0);

    if (wireframe) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    materialToUse->unbind();
}
