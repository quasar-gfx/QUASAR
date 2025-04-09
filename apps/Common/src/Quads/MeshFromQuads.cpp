#include <Quads/MeshFromQuads.h>

using namespace quasar;

MeshFromQuads::MeshFromQuads(glm::uvec2 &remoteWindowSize, unsigned int maxNumProxies)
        : remoteWindowSize(remoteWindowSize)
        , depthBufferSize(2u * remoteWindowSize) // 4 offsets per pixel
        , maxProxies(maxNumProxies)
        , meshSizesBuffer(GL_SHADER_STORAGE_BUFFER, 1, sizeof(BufferSizes), nullptr, GL_DYNAMIC_COPY)
        , prevNumProxiesBuffer(GL_SHADER_STORAGE_BUFFER, 1, sizeof(unsigned int), nullptr, GL_DYNAMIC_DRAW)
        , currNumProxiesBuffer(GL_SHADER_STORAGE_BUFFER, 1, sizeof(unsigned int), nullptr, GL_DYNAMIC_DRAW)
        , currentQuadBuffers(maxProxies)
        , quadIndicesBuffer(GL_SHADER_STORAGE_BUFFER, remoteWindowSize.x * remoteWindowSize.y, sizeof(unsigned int), nullptr, GL_DYNAMIC_DRAW)
        , quadCreatedFlagsBuffer(GL_SHADER_STORAGE_BUFFER, maxProxies, sizeof(int), nullptr, GL_DYNAMIC_DRAW)
        , appendProxiesShader({
            .computeCodePath = "shaders/appendProxies.comp",
            .defines = {
                "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
            }
        })
        , fillQuadIndicesShader({
            .computeCodePath = "shaders/fillQuadIndices.comp",
            .defines = {
                "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
            }
        })
        , createMeshFromQuadsShader({
            .computeCodePath = "shaders/createMeshFromQuads.comp",
            .defines = {
                "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
            }
        }) {
}

MeshFromQuads::BufferSizes MeshFromQuads::getBufferSizes() {
    BufferSizes bufferSizes;

    meshSizesBuffer.bind();
    meshSizesBuffer.getData(&bufferSizes);
    return bufferSizes;
}

void MeshFromQuads::appendProxies(
        const glm::uvec2 &gBufferSize,
        unsigned int numProxies,
        const QuadBuffers &newQuadBuffers,
        bool iFrame) {
    double startTime = timeutils::getTimeMicros();

    appendProxiesShader.bind();
    {
        appendProxiesShader.setBool("iFrame", iFrame);
        appendProxiesShader.setUint("newNumProxies", numProxies);
    }
    {
        appendProxiesShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, currNumProxiesBuffer);
        appendProxiesShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, prevNumProxiesBuffer);

        appendProxiesShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 2, newQuadBuffers.normalSphericalsBuffer);
        appendProxiesShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 3, newQuadBuffers.depthsBuffer);
        appendProxiesShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 4, newQuadBuffers.offsetSizeFlattenedsBuffer);

        appendProxiesShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 5, currentQuadBuffers.normalSphericalsBuffer);
        appendProxiesShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 6, currentQuadBuffers.depthsBuffer);
        appendProxiesShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 7, currentQuadBuffers.offsetSizeFlattenedsBuffer);
    }
    appendProxiesShader.dispatch(((numProxies + 1) + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1, 1);
    appendProxiesShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    stats.timeToAppendProxiesMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    fillQuadIndices(gBufferSize);
}

void MeshFromQuads::fillQuadIndices(const glm::uvec2 &gBufferSize) {
    double startTime = timeutils::getTimeMicros();

    fillQuadIndicesShader.bind();
    {
        fillQuadIndicesShader.setVec2("remoteWindowSize", gBufferSize);
    }
    {
        fillQuadIndicesShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, currNumProxiesBuffer);
        fillQuadIndicesShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, prevNumProxiesBuffer);
        fillQuadIndicesShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 2, quadCreatedFlagsBuffer);

        fillQuadIndicesShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 3, currentQuadBuffers.normalSphericalsBuffer);
        fillQuadIndicesShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 4, currentQuadBuffers.depthsBuffer);
        fillQuadIndicesShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 5, currentQuadBuffers.offsetSizeFlattenedsBuffer);

        fillQuadIndicesShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 6, quadIndicesBuffer);
    }
    fillQuadIndicesShader.dispatch((MAX_NUM_PROXIES + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1, 1);
    fillQuadIndicesShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    stats.timeToFillOutputQuadsMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
}

void MeshFromQuads::createMeshFromProxies(
        const glm::uvec2 &gBufferSize,
        unsigned int numProxies,
        const DepthOffsets &depthOffsets,
        const PerspectiveCamera &remoteCamera,
        const Mesh &mesh) {
    double startTime = timeutils::getTimeMicros();

    createMeshFromQuadsShader.bind();
    {
        createMeshFromQuadsShader.setVec2("remoteWindowSize", gBufferSize);
        createMeshFromQuadsShader.setVec2("depthBufferSize", depthBufferSize);
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
        createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 7, currentQuadBuffers.offsetSizeFlattenedsBuffer);

        createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 8, quadIndicesBuffer);

        createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 9, currNumProxiesBuffer);

        createMeshFromQuadsShader.setImageTexture(0, depthOffsets.buffer, 0, GL_FALSE, 0, GL_READ_ONLY, depthOffsets.buffer.internalFormat);
    }
    createMeshFromQuadsShader.dispatch((gBufferSize.x + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                       (gBufferSize.y + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
    createMeshFromQuadsShader.memoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);

    stats.timeToCreateMeshMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
}
