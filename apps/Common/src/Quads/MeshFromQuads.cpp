#include <Quads/MeshFromQuads.h>

MeshFromQuads::MeshFromQuads(const glm::uvec2 &remoteWindowSize)
        : remoteWindowSize(remoteWindowSize)
        , depthBufferSize(2u * remoteWindowSize) // 4 offsets per pixel
        , maxProxies(MAX_NUM_PROXIES)
        , meshSizesBuffer(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_COPY, 1, nullptr)
        , quadIndicesBuffer({
            .width = remoteWindowSize.x,
            .height = remoteWindowSize.y,
            .internalFormat = GL_R32UI,
            .format = GL_RED_INTEGER,
            .type = GL_UNSIGNED_INT,
            .wrapS = GL_CLAMP_TO_EDGE,
            .wrapT = GL_CLAMP_TO_EDGE,
            .minFilter = GL_NEAREST,
            .magFilter = GL_NEAREST
        })
        , currNumProxiesBuffer(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_COPY, 1, nullptr)
        , currentQuadBuffers(maxProxies)
        , quadCreatedFlagsBuffer(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW, maxProxies, nullptr)
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
    createMeshFromQuadsShader.bind();
    createMeshFromQuadsShader.setVec2("remoteWindowSize", remoteWindowSize);

    fillQuadIndicesShader.bind();
    fillQuadIndicesShader.setVec2("remoteWindowSize", remoteWindowSize);
}

MeshFromQuads::BufferSizes MeshFromQuads::getBufferSizes() {
    BufferSizes bufferSizes;

    meshSizesBuffer.bind();
    meshSizesBuffer.getData(&bufferSizes);
    return bufferSizes;
}

void MeshFromQuads::appendProxies(
        unsigned int numProxies,
        const QuadBuffers &newQuadBuffers,
        bool iFrame) {
    appendProxiesShader.startTiming();

    appendProxiesShader.bind();
    {
        appendProxiesShader.setBool("iFrame", iFrame);
        appendProxiesShader.setUint("numNewProxies", numProxies);
        appendProxiesShader.setUint("prevNumProxies", currNumProxies);
    }
    {
        appendProxiesShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, currNumProxiesBuffer);
        appendProxiesShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, quadCreatedFlagsBuffer);

        appendProxiesShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 2, newQuadBuffers.normalSphericalsBuffer);
        appendProxiesShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 3, newQuadBuffers.depthsBuffer);
        appendProxiesShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 4, newQuadBuffers.offsetSizeFlattenedsBuffer);

        appendProxiesShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 5, currentQuadBuffers.normalSphericalsBuffer);
        appendProxiesShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 6, currentQuadBuffers.depthsBuffer);
        appendProxiesShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 7, currentQuadBuffers.offsetSizeFlattenedsBuffer);
    }
    appendProxiesShader.dispatch((numProxies + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1, 1);
    appendProxiesShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    currNumProxiesBuffer.bind();
    currNumProxiesBuffer.getData(&currNumProxies);

    appendProxiesShader.endTiming();
    stats.timeToAppendProxiesMs = appendProxiesShader.getElapsedTime();

    fillQuadIndices();
}

void MeshFromQuads::fillQuadIndices() {
    fillQuadIndicesShader.startTiming();

    fillQuadIndicesShader.bind();
    {
        fillQuadIndicesShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, currNumProxiesBuffer);
        fillQuadIndicesShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, quadCreatedFlagsBuffer);

        fillQuadIndicesShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 2, currentQuadBuffers.normalSphericalsBuffer);
        fillQuadIndicesShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 3, currentQuadBuffers.depthsBuffer);
        fillQuadIndicesShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 4, currentQuadBuffers.offsetSizeFlattenedsBuffer);

        fillQuadIndicesShader.setImageTexture(0, quadIndicesBuffer, 0, GL_FALSE, 0, GL_WRITE_ONLY, quadIndicesBuffer.internalFormat);
    }
    fillQuadIndicesShader.dispatch((currNumProxies + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1, 1);
    fillQuadIndicesShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    fillQuadIndicesShader.endTiming();
    stats.timeToFillOutputQuadsMs = fillQuadIndicesShader.getElapsedTime();
}

void MeshFromQuads::createMeshFromProxies(
        unsigned int numProxies,
        const DepthOffsets &depthOffsets,
        const PerspectiveCamera &remoteCamera,
        const Mesh& mesh) {
    createMeshFromQuadsShader.bind();
    createMeshFromQuadsShader.setImageTexture(0, depthOffsets.buffer, 0, GL_FALSE, 0, GL_READ_ONLY, depthOffsets.buffer.internalFormat);

    createMeshFromProxies(numProxies, remoteCamera, mesh, false);
}

void MeshFromQuads::appendGeometry(
        unsigned int numProxies,
        const DepthOffsets &depthOffsets,
        const PerspectiveCamera &remoteCamera,
        const Mesh &mesh) {
    createMeshFromQuadsShader.bind();
    createMeshFromQuadsShader.setImageTexture(0, depthOffsets.buffer, 0, GL_FALSE, 0, GL_READ_ONLY, depthOffsets.buffer.internalFormat);

    createMeshFromProxies(numProxies, remoteCamera, mesh, true);
}

void MeshFromQuads::createMeshFromProxies(
        unsigned int numProxies,
        const PerspectiveCamera &remoteCamera,
        const Mesh &mesh,
        bool appendGeometry) {
    createMeshFromQuadsShader.startTiming();

    createMeshFromQuadsShader.bind();
    {
        createMeshFromQuadsShader.setMat4("view", remoteCamera.getViewMatrix());
        createMeshFromQuadsShader.setMat4("projection", remoteCamera.getProjectionMatrix());
        createMeshFromQuadsShader.setMat4("viewInverse", remoteCamera.getViewMatrixInverse());
        createMeshFromQuadsShader.setMat4("projectionInverse", remoteCamera.getProjectionMatrixInverse());
        createMeshFromQuadsShader.setFloat("near", remoteCamera.getNear());
        createMeshFromQuadsShader.setFloat("far", remoteCamera.getFar());
    }
    {
        createMeshFromQuadsShader.setBool("appendGeometry", appendGeometry);
        createMeshFromQuadsShader.setUint("quadMapSize", numProxies);
        createMeshFromQuadsShader.setVec2("depthBufferSize", depthBufferSize);
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

        fillQuadIndicesShader.setImageTexture(1, quadIndicesBuffer, 0, GL_FALSE, 0, GL_WRITE_ONLY, quadIndicesBuffer.internalFormat);
    }
    createMeshFromQuadsShader.dispatch((remoteWindowSize.x + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                       (remoteWindowSize.y + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
    createMeshFromQuadsShader.memoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);

    createMeshFromQuadsShader.endTiming();
    stats.timeToCreateMeshMs = createMeshFromQuadsShader.getElapsedTime();
}
