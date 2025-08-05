#include <Quads/MeshFromQuads.h>
#include <Utils/TimeUtils.h>

#include <shaders_common.h>

using namespace quasar;

MeshFromQuads::MeshFromQuads(const QuadFrame& quadFrame, uint maxNumProxies)
    : maxProxies(maxNumProxies)
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
        .numElems = quadFrame.getSize().x * quadFrame.getSize().y,
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
    , createMeshFromQuadsShader({
        .computeCodeData = SHADER_COMMON_MESH_FROM_QUADS_COMP,
        .computeCodeSize = SHADER_COMMON_MESH_FROM_QUADS_COMP_len,
        .defines = {
            "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
        }
    })
{}

MeshFromQuads::BufferSizes MeshFromQuads::getBufferSizes() const {
    BufferSizes bufferSizes;

    meshSizesBuffer.bind();
    meshSizesBuffer.getData(&bufferSizes);
    return bufferSizes;
}

void MeshFromQuads::appendQuads(const QuadFrame& quadFrame, const glm::vec2& gBufferSize, bool isRefFrame) {
    appendQuadsShader.startTiming();

    appendQuadsShader.bind();
    {
        appendQuadsShader.setBool("isRefFrame", isRefFrame);
        appendQuadsShader.setUint("newNumProxies", quadFrame.getNumQuads());
    }
    {
        appendQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, currNumProxiesBuffer);
        appendQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, prevNumProxiesBuffer);

        appendQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 2, quadFrame.quadBuffers.normalSphericalsBuffer);
        appendQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 3, quadFrame.quadBuffers.depthsBuffer);
        appendQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 4, quadFrame.quadBuffers.metadatasBuffer);

        appendQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 5, currentQuadBuffers.normalSphericalsBuffer);
        appendQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 6, currentQuadBuffers.depthsBuffer);
        appendQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 7, currentQuadBuffers.metadatasBuffer);
    }
    appendQuadsShader.dispatch(((quadFrame.getNumQuads() + 1) + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1, 1);
    appendQuadsShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    appendQuadsShader.endTiming();
    stats.timeToAppendQuadsMs = appendQuadsShader.getElapsedTime();

    fillQuadIndices(quadFrame, gBufferSize);
}

void MeshFromQuads::fillQuadIndices(const QuadFrame& quadFrame, const glm::vec2& gBufferSize) {
    fillQuadIndicesShader.startTiming();

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
    fillQuadIndicesShader.dispatch((MAX_NUM_PROXIES + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1, 1);
    fillQuadIndicesShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    fillQuadIndicesShader.endTiming();
    stats.timeToGatherQuadsMs = fillQuadIndicesShader.getElapsedTime();
}

void MeshFromQuads::createMeshFromProxies(const QuadFrame& quadFrame, const glm::vec2& gBufferSize, const PerspectiveCamera& remoteCamera, const Mesh& mesh) {
    createMeshFromQuadsShader.startTiming();

    createMeshFromQuadsShader.bind();
    {
        createMeshFromQuadsShader.setVec2("gBufferSize", gBufferSize);
    }
    {
        createMeshFromQuadsShader.setMat4("view", remoteCamera.getViewMatrix());
        createMeshFromQuadsShader.setMat4("projection", remoteCamera.getProjectionMatrix());
        createMeshFromQuadsShader.setMat4("viewInverse", remoteCamera.getViewMatrixInverse());
        createMeshFromQuadsShader.setMat4("projectionInverse", remoteCamera.getProjectionMatrixInverse());
        createMeshFromQuadsShader.setFloat("near", remoteCamera.getNear());
        createMeshFromQuadsShader.setFloat("far", remoteCamera.getFar());
    }
    {
        createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, meshSizesBuffer);
        createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, quadCreatedFlagsBuffer);

        createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 2, mesh.vertexBuffer);
        createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 3, mesh.indexBuffer);
        createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 4, mesh.indirectBuffer);

        createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 5, currentQuadBuffers.normalSphericalsBuffer);
        createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 6, currentQuadBuffers.depthsBuffer);
        createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 7, currentQuadBuffers.metadatasBuffer);

        createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 8, quadIndicesMap);

        createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 9, currNumProxiesBuffer);

        createMeshFromQuadsShader.setImageTexture(0, quadFrame.depthOffsets.buffer, 0, GL_FALSE, 0, GL_READ_ONLY, quadFrame.depthOffsets.buffer.internalFormat);
    }
    createMeshFromQuadsShader.dispatch((quadFrame.getSize().x + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                       (quadFrame.getSize().y + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
    createMeshFromQuadsShader.memoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);

    createMeshFromQuadsShader.endTiming();
    stats.timeToCreateMeshMs = createMeshFromQuadsShader.getElapsedTime();
}
