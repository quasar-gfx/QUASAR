#include <Primitives/Mesh.h>

#include <glm/gtx/string_cast.hpp>

using namespace quasar;

Mesh::Mesh()
    : vertexBuffer(GL_ARRAY_BUFFER, sizeof(Vertex))
    , indexBuffer(GL_ELEMENT_ARRAY_BUFFER, sizeof(uint))
{
    setArrayBufferAttributes(Vertex::getVertexInputAttributes(), sizeof(Vertex));
}

Mesh::Mesh(const MeshDataCreateParams& params)
    : material(params.material)
    , IBL(params.IBL)
    , usage(params.usage)
    , vertexBuffer(GL_ARRAY_BUFFER, sizeof(Vertex), params.usage)
    , indexBuffer(GL_ELEMENT_ARRAY_BUFFER, sizeof(uint), params.usage)
    , indirectDraw(params.indirectDraw)
    , indirectBuffer(GL_DRAW_INDIRECT_BUFFER, sizeof(DrawElementsIndirectCommand), params.usage)
    , vertexSize(params.vertexSize)
    , attributes(params.attributes)
{
    setArrayBufferAttributes(params.attributes, params.vertexSize);
    setBuffers(params.verticesData, params.verticesSize, params.indicesData, params.indicesSize);

    if (indirectDraw) {
        indirectBuffer.bind();
        DrawElementsIndirectCommand indirectCommand;
        indirectBuffer.setData(sizeof(DrawElementsIndirectCommand), &indirectCommand);
        indirectBuffer.unbind();
    }
}

Mesh::Mesh(const MeshSizeCreateParams& params)
    : material(params.material)
    , IBL(params.IBL)
    , usage(params.usage)
    , vertexBuffer(GL_ARRAY_BUFFER, sizeof(Vertex), params.usage)
    , indexBuffer(GL_ELEMENT_ARRAY_BUFFER, sizeof(uint), params.usage)
    , indirectDraw(params.indirectDraw)
    , indirectBuffer(GL_DRAW_INDIRECT_BUFFER, sizeof(DrawElementsIndirectCommand), params.usage)
    , vertexSize(params.vertexSize)
    , attributes(params.attributes)
{
    setArrayBufferAttributes(params.attributes, params.vertexSize);
    setBuffers(params.maxVertices, params.maxIndices);

    if (indirectDraw) {
        indirectBuffer.bind();
        DrawElementsIndirectCommand indirectCommand;
        indirectBuffer.setData(sizeof(DrawElementsIndirectCommand), &indirectCommand);
        indirectBuffer.unbind();
    }
}

void Mesh::setArrayBufferAttributes(const VertexInputAttributes& attributes, uint vertexSize) {
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

void Mesh::setBuffers(const void* verticesData, uint verticesSize, const uint* indicesData, uint indicesSize) {
    // If no vertices, dont bind buffers
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

void Mesh::setBuffers(uint verticesSize, uint indicesSize) {
    // If no vertices or indices, dont bind buffers
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

void Mesh::resizeBuffers(uint verticesSize, uint indicesSize) {
    vertexBuffer.resize(verticesSize);
    indexBuffer.resize(indicesSize);
}

void Mesh::updateAABB(const void* verticesData, uint verticesSize) {
    // If no vertices, return
    if (verticesData == nullptr || verticesSize == 0) {
        return;
    }

    auto* verticesVec = reinterpret_cast<const Vertex*>(verticesData);
    glm::vec3 min = verticesVec[0].position;
    glm::vec3 max = verticesVec[0].position;

    for (uint i = 1; i < verticesSize; i++) {
        auto& vertex = verticesVec[i];
        min = glm::min(min, vertex.position);
        max = glm::max(max, vertex.position);
    }

    // Set up AABB
    aabb.update(min, max);
}

void Mesh::setMaterialCameraParams(const Camera& camera, const Material* material) {
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

void Mesh::bindMaterial(Scene& scene, const glm::mat4& model, const Material* overrideMaterial, const Texture* prevIDMap) {
    auto* materialToUse = overrideMaterial != nullptr ? overrideMaterial : material;
    materialToUse->bind();

    auto* shader = materialToUse->getShader();

    // Update material uniforms with lighting information
    int texIdx = scene.bindMaterial(materialToUse);
    shader->setFloat("material.IBL", IBL);

    shader->setBool("peelDepth", prevIDMap != nullptr);
    if (prevIDMap != nullptr) {
        shader->setTexture("prevIDMap", *prevIDMap, texIdx);
    }

    materialToUse->unbind();
}

RenderStats Mesh::draw(GLenum primativeType, const Camera& camera, const glm::mat4& model, bool frustumCull, const Material* overrideMaterial) {
    RenderStats stats;

    // If the camera is a VR camera, check if the AABB is visible in both frustums
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

    // Set draw ID/object ID
    materialToUse->getShader()->setUint("drawID", ID);

    // Set camera params
    setMaterialCameraParams(camera, materialToUse);

    // Set model and normal matrix
    materialToUse->getShader()->setMat4("model", model);
    materialToUse->getShader()->setMat3("normalMatrix", glm::transpose(glm::inverse(glm::mat3(model))));

    stats = draw(primativeType);

    materialToUse->unbind();

    return stats;
}

RenderStats Mesh::draw(GLenum primativeType, const Camera& camera, const glm::mat4& model, const BoundingSphere& boundingSphere, const Material* overrideMaterial) {
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
        stats.trianglesDrawn = static_cast<uint>(indexBuffer.getSize() / 3);
    }
    else {
        stats.trianglesDrawn = static_cast<uint>(vertexBuffer.getSize() / 3);
    }
    stats.drawCalls = 1;

    return stats;
}
