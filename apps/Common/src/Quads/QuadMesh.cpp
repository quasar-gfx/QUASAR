#include <Quads/QuadMesh.h>
#include <Utils/TimeUtils.h>
#include <shaders_common.h>

#ifndef PLATFORM_ANDROID
#define THREADS_PER_LOCALGROUP 32
#else
#define THREADS_PER_LOCALGROUP 16
#endif

using namespace quasar;

QuadMesh::QuadMesh(const QuadSet& quadSet, Texture& colorTexture, Texture& alphaTexture, const glm::vec4& textureExtent, uint maxProxies)
    : maxProxies(maxProxies)
    , currentQuadBuffers(maxProxies)
    , textureExtent(textureExtent)
    , sizesBuffer({
        .target = GL_SHADER_STORAGE_BUFFER,
        .dataSize = sizeof(BufferSizes),
        .numElems = 1,
        .usage = GL_DYNAMIC_COPY,
    })
    , quadIndexMap({
        .target = GL_SHADER_STORAGE_BUFFER,
        .dataSize = sizeof(uint32_t),
        .numElems = quadSet.getSize().x * quadSet.getSize().y,
        .usage = GL_DYNAMIC_DRAW,
    })
    , quadCreatedFlags({
        .target = GL_SHADER_STORAGE_BUFFER,
        .dataSize = sizeof(int),
        .numElems = maxProxies,
        .usage = GL_DYNAMIC_DRAW,
    })
    , appendQuadsShader({
        .computeCodeData = SHADER_COMMON_APPEND_QUADS_COMP,
        .computeCodeSize = SHADER_COMMON_APPEND_QUADS_COMP_len,
        .defines = {
            "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
        }
    })
    , createQuadMeshShader({
        .computeCodeData = SHADER_COMMON_MESH_FROM_QUADS_COMP,
        .computeCodeSize = SHADER_COMMON_MESH_FROM_QUADS_COMP_len,
        .defines = {
            "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
        }
    })
    , Mesh({
        .maxVertices = 0,
        .maxIndices = 0,
        .vertexSize = sizeof(QuadVertex),
        .attributes = QuadVertex::getVertexInputAttributes(),
        .material = new QuadMaterial({ .baseColorTexture = &colorTexture, .alphaTexture = &alphaTexture }),
        .usage = GL_DYNAMIC_DRAW,
        .indirectDraw = true
    })
    , indexBufferTransparent({
        .target = GL_ELEMENT_ARRAY_BUFFER,
        .dataSize = sizeof(uint),
        .usage = GL_DYNAMIC_DRAW,
    })
    , indirectBufferTransparent({
        .target = GL_DRAW_INDIRECT_BUFFER,
        .dataSize = sizeof(DrawElementsIndirectCommand),
        .usage = GL_DYNAMIC_DRAW,
    })
{
    indirectBufferTransparent.bind();
    DrawElementsIndirectCommand indirectCommand;
    indirectBufferTransparent.setData(sizeof(DrawElementsIndirectCommand), &indirectCommand);
    indirectBufferTransparent.unbind();
}

QuadMesh::QuadMesh(const QuadSet& quadSet, Texture& colorTexture, Texture& alphaTexture, uint maxProxies)
    : QuadMesh(quadSet, colorTexture, alphaTexture, glm::vec4(0.0f, 0.0f, 1.0f, 1.0f), maxProxies)
{}

QuadMesh::BufferSizes QuadMesh::getBufferSizes() const {
    BufferSizes bufferSizes;
    sizesBuffer.bind();
    void* ptr = sizesBuffer.mapToCPU(GL_MAP_READ_BIT);
    if (ptr) {
        std::memcpy(&bufferSizes, ptr, sizeof(BufferSizes));
        sizesBuffer.unmapFromCPU();
    }
    else {
        spdlog::warn("Failed to map sizesBuffer. Copying using getData");
        sizesBuffer.getData(&bufferSizes);
    }
    return bufferSizes;
}

void QuadMesh::appendQuads(const QuadSet& quadSet, const glm::vec2& gBufferSize, bool isFullFrame) {
    double startTime = timeutils::getTimeMicros();

    if (isFullFrame) {
        // Clear current proxy count
        currNumProxies = 0;
    }

    uint incomingNumProxies = quadSet.getNumProxies();
    uint incomingNumProxiesTransparent = quadSet.getNumProxiesTransparent();
    uint newNumProxies = currNumProxies + incomingNumProxies;
    if (newNumProxies > maxProxies) {
        spdlog::warn("Max proxies exceeded! Clamping {} to {} proxies.", newNumProxies, maxProxies);
        newNumProxies = maxProxies;
    }

    appendQuadsShader.bind();
    {
        appendQuadsShader.setVec2("gBufferSize", gBufferSize);
        appendQuadsShader.setUint("currNumProxies", currNumProxies);
        appendQuadsShader.setUint("newNumProxies", newNumProxies);
    }
    {
        appendQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, quadSet.quadBuffers.normalAndDepthBuffer);
        appendQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, quadSet.quadBuffers.metadatasBuffer);

        appendQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 2, currentQuadBuffers.normalAndDepthBuffer);
        appendQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 3, currentQuadBuffers.metadatasBuffer);

        appendQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 4, quadCreatedFlags);
        appendQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 5, quadIndexMap);
    }
    appendQuadsShader.dispatch((newNumProxies + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1, 1);
    appendQuadsShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    // Set new current proxy count
    currNumProxies = newNumProxies;
    currNumProxiesTransparent = incomingNumProxiesTransparent;

    stats.appendQuadsTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
}

void QuadMesh::createMeshFromProxies(const QuadSet& quadSet, const glm::vec2& gBufferSize, const PerspectiveCamera& remoteCamera) {
    double startTime = timeutils::getTimeMicros();

    // Resize buffers if more space is needed
    size_t newNumVertices = currNumProxies * NUM_SUB_QUADS * VERTICES_IN_A_QUAD;
    if (vertexBuffer.getSize() < newNumVertices) {
        vertexBuffer.bind();
        vertexBuffer.smartResize(newNumVertices, false);
    }
    size_t numProxiesOpaque = currNumProxies - currNumProxiesTransparent;
    size_t newNumIndices = numProxiesOpaque * NUM_SUB_QUADS * INDICES_IN_A_QUAD;
    if (indexBuffer.getSize() < newNumIndices) {
        indexBuffer.bind();
        indexBuffer.smartResize(newNumIndices, false);
    }
    size_t newNumIndicesTransparent = currNumProxiesTransparent * NUM_SUB_QUADS * INDICES_IN_A_QUAD;
    if (indexBufferTransparent.getSize() < newNumIndicesTransparent) {
        indexBufferTransparent.bind();
        indexBufferTransparent.smartResize(newNumIndicesTransparent, false);
    }

    createQuadMeshShader.bind();
    {
        createQuadMeshShader.setVec2("gBufferSize", gBufferSize);
        createQuadMeshShader.setUint("currNumProxies", currNumProxies);
        createQuadMeshShader.setVec4("textureExtent", textureExtent);
        createQuadMeshShader.setFloat("expandQuadAmount", expandQuadAmount);
    }
    {
        createQuadMeshShader.setMat4("view", remoteCamera.getViewMatrix());
        createQuadMeshShader.setMat4("projection", remoteCamera.getProjectionMatrix());
        createQuadMeshShader.setMat4("viewInverse", remoteCamera.getViewMatrixInverse());
        createQuadMeshShader.setMat4("projectionInverse", remoteCamera.getProjectionMatrixInverse());
        createQuadMeshShader.setFloat("near", remoteCamera.getNear());
        createQuadMeshShader.setFloat("far", remoteCamera.getFar());
    }
    {
        createQuadMeshShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, sizesBuffer);

        createQuadMeshShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, vertexBuffer);
        createQuadMeshShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 2, indexBuffer);
        createQuadMeshShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 3, indexBufferTransparent);
        createQuadMeshShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 4, indirectBuffer);
        createQuadMeshShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 5, indirectBufferTransparent);

        createQuadMeshShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 6, currentQuadBuffers.normalAndDepthBuffer);
        createQuadMeshShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 7, currentQuadBuffers.metadatasBuffer);

        createQuadMeshShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 8, quadCreatedFlags);
        createQuadMeshShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 9, quadIndexMap);

        createQuadMeshShader.setImageTexture(0, quadSet.depthOffsets.texture, 0, GL_FALSE, 0, GL_READ_ONLY, quadSet.depthOffsets.texture.internalFormat);
    }
    createQuadMeshShader.dispatch((quadSet.getSize().x + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                  (quadSet.getSize().y + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
    createQuadMeshShader.memoryBarrier(
        GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT |
        GL_COMMAND_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);

    stats.createMeshTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
}

RenderStats QuadMesh::draw(GLenum primitiveType) {
    RenderStats stats;

    BufferSizes bufferSizes = getBufferSizes();

    glBindVertexArray(vertexArrayBuffer);
    // Draw opaque proxies first
    if ((drawState == DrawState::OPAQUE || drawState == DrawState::BOTH) && indexBuffer.getSize() > 0) {
        indirectBuffer.bind();
        indexBuffer.bind();
        glDrawElementsIndirect(primitiveType, GL_UNSIGNED_INT, 0);
        indexBuffer.unbind();

        stats.trianglesDrawn = static_cast<uint>(bufferSizes.numIndices / 3);
    }
    // Draw transparent proxies after opaque ones
    if ((drawState == DrawState::TRANSPARENT || drawState == DrawState::BOTH) && indexBufferTransparent.getSize() > 0) {
        indirectBufferTransparent.bind();
        indexBufferTransparent.bind();
        glDrawElementsIndirect(primitiveType, GL_UNSIGNED_INT, 0);
        indexBufferTransparent.unbind();

        stats.trianglesDrawn = static_cast<uint>(bufferSizes.numIndicesTransparent / 3);
    }
    glBindVertexArray(0);

    stats.drawCalls = 1;

    return stats;
}
