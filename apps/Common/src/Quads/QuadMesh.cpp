#include <Quads/QuadMesh.h>
#include <Utils/TimeUtils.h>

#include <shaders_common.h>

#ifndef __ANDROID__
#define THREADS_PER_LOCALGROUP 32
#else
#define THREADS_PER_LOCALGROUP 16
#endif

using namespace quasar;

QuadMesh::QuadMesh(const QuadSet& quadSet, Texture& colorTexture, uint maxQuadsPerMesh)
    : maxProxies(maxQuadsPerMesh)
    , currentQuadBuffers(maxProxies)
    , meshSizesBuffer({
        .target = GL_SHADER_STORAGE_BUFFER,
        .dataSize = sizeof(BufferSizes),
        .numElems = 1,
        .usage = GL_DYNAMIC_COPY,
    })
    , prevNumProxiesBuffer({
        .target = GL_SHADER_STORAGE_BUFFER,
        .dataSize = sizeof(uint),
        .numElems = 1,
        .usage = GL_DYNAMIC_DRAW,
    })
    , currNumProxiesBuffer({
        .target = GL_SHADER_STORAGE_BUFFER,
        .dataSize = sizeof(uint),
        .numElems = 1,
        .usage = GL_DYNAMIC_DRAW,
    })
    , quadIndicesMap({
        .target = GL_SHADER_STORAGE_BUFFER,
        .dataSize = sizeof(uint),
        .numElems = quadSet.getSize().x * quadSet.getSize().y,
        .usage = GL_DYNAMIC_DRAW,
    })
    , quadCreatedFlagsBuffer({
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
    , fillQuadIndicesShader({
        .computeCodeData = SHADER_COMMON_FILL_QUAD_INDICES_COMP,
        .computeCodeSize = SHADER_COMMON_FILL_QUAD_INDICES_COMP_len,
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
        .maxVertices = maxQuadsPerMesh * NUM_SUB_QUADS * VERTICES_IN_A_QUAD,
        .maxIndices = maxQuadsPerMesh * NUM_SUB_QUADS * INDICES_IN_A_QUAD,
        .vertexSize = sizeof(QuadVertex),
        .attributes = QuadVertex::getVertexInputAttributes(),
        .material = new QuadMaterial({ .baseColorTexture = &colorTexture }),
        .usage = GL_DYNAMIC_DRAW,
        .indirectDraw = true
    })
{}

QuadMesh::BufferSizes QuadMesh::getBufferSizes() const {
    BufferSizes bufferSizes;
    meshSizesBuffer.bind();
    meshSizesBuffer.getData(&bufferSizes);
    meshSizesBuffer.unbind();
    return bufferSizes;
}

void QuadMesh::appendQuads(const QuadSet& quadSet, const glm::vec2& gBufferSize, bool isRefFrame) {
    double startTime = timeutils::getTimeMicros();

    appendQuadsShader.bind();
    {
        appendQuadsShader.setBool("isRefFrame", isRefFrame);
        appendQuadsShader.setUint("newNumProxies", quadSet.getNumQuads());
    }
    {
        appendQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, currNumProxiesBuffer);
        appendQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, prevNumProxiesBuffer);

        appendQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 2, quadSet.quadBuffers.normalSphericalsBuffer);
        appendQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 3, quadSet.quadBuffers.depthsBuffer);
        appendQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 4, quadSet.quadBuffers.metadatasBuffer);

        appendQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 5, currentQuadBuffers.normalSphericalsBuffer);
        appendQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 6, currentQuadBuffers.depthsBuffer);
        appendQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 7, currentQuadBuffers.metadatasBuffer);
    }
    appendQuadsShader.dispatch(((quadSet.getNumQuads() + 1) + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1, 1);
    appendQuadsShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    stats.timeToAppendQuadsMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    fillQuadIndices(quadSet, gBufferSize);
}

void QuadMesh::fillQuadIndices(const QuadSet& quadSet, const glm::vec2& gBufferSize) {
    double startTime = timeutils::getTimeMicros();

    fillQuadIndicesShader.bind();
    {
        fillQuadIndicesShader.setVec2("gBufferSize", gBufferSize);
    }
    {
        fillQuadIndicesShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, currNumProxiesBuffer);
        fillQuadIndicesShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, prevNumProxiesBuffer);
        fillQuadIndicesShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 2, quadCreatedFlagsBuffer);

        fillQuadIndicesShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 3, currentQuadBuffers.normalSphericalsBuffer);
        fillQuadIndicesShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 4, currentQuadBuffers.depthsBuffer);
        fillQuadIndicesShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 5, currentQuadBuffers.metadatasBuffer);

        fillQuadIndicesShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 6, quadIndicesMap);
    }
    fillQuadIndicesShader.dispatch((MAX_QUADS_PER_MESH + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1, 1);
    fillQuadIndicesShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    stats.timeToGatherQuadsMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
}

void QuadMesh::createMeshFromProxies(const QuadSet& quadSet, const glm::vec2& gBufferSize, const PerspectiveCamera& remoteCamera) {
    double startTime = timeutils::getTimeMicros();

    createQuadMeshShader.bind();
    {
        createQuadMeshShader.setVec2("gBufferSize", gBufferSize);
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
        createQuadMeshShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, meshSizesBuffer);
        createQuadMeshShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, quadCreatedFlagsBuffer);

        createQuadMeshShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 2, vertexBuffer);
        createQuadMeshShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 3, indexBuffer);
        createQuadMeshShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 4, indirectBuffer);

        createQuadMeshShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 5, currentQuadBuffers.normalSphericalsBuffer);
        createQuadMeshShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 6, currentQuadBuffers.depthsBuffer);
        createQuadMeshShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 7, currentQuadBuffers.metadatasBuffer);

        createQuadMeshShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 8, quadIndicesMap);

        createQuadMeshShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 9, currNumProxiesBuffer);

        createQuadMeshShader.setImageTexture(0, quadSet.depthOffsets.texture, 0, GL_FALSE, 0, GL_READ_ONLY, quadSet.depthOffsets.texture.internalFormat);
    }
    createQuadMeshShader.dispatch((quadSet.getSize().x + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                  (quadSet.getSize().y + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
    createQuadMeshShader.memoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);

    stats.timeToCreateMeshMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
}
