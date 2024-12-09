#include <MeshFromQuads.h>

MeshFromQuads::MeshFromQuads(const glm::uvec2 &remoteWindowSize)
        : remoteWindowSize(remoteWindowSize)
        , atlasSize(4u * remoteWindowSize)
        , sizesBuffer(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_COPY, 1, nullptr)
        , atlasOffsetBuffer(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_COPY, 1, nullptr)
        , atlas({
            .width = atlasSize.x,
            .height = atlasSize.y,
            .internalFormat = GL_RGBA16F,
            .format = GL_RGBA,
            .type = GL_HALF_FLOAT,
            .wrapS = GL_CLAMP_TO_EDGE,
            .wrapT = GL_CLAMP_TO_EDGE,
            .minFilter = GL_NEAREST,
            .magFilter = GL_NEAREST
        })
        , createMeshFromQuadsShader({
            .computeCodePath = "shaders/createMeshFromQuads.comp",
            .defines = {
                "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
            }
        }) {
    createMeshFromQuadsShader.bind();
    createMeshFromQuadsShader.setVec2("remoteWindowSize", remoteWindowSize);
    createMeshFromQuadsShader.setVec2("atlasSize", atlasSize);
}

MeshFromQuads::BufferSizes MeshFromQuads::getBufferSizes() {
    BufferSizes bufferSizes;

    sizesBuffer.bind();
    sizesBuffer.getData(&bufferSizes);
    return bufferSizes;
}

void MeshFromQuads::createMeshFromProxies(
        unsigned int numProxies, const glm::uvec2 &depthBufferSize,
        const PerspectiveCamera &remoteCamera,
        const QuadBuffers &quadBuffers,
        const Texture& depthOffsetsBuffer,
        const Texture& colorTexture,
        const Mesh& mesh) {
    createMeshFromQuadsShader.bind();
    createMeshFromQuadsShader.setImageTexture(0, depthOffsetsBuffer, 0, GL_FALSE, 0, GL_READ_ONLY, depthOffsetsBuffer.internalFormat);

    createMeshFromProxies(
            numProxies, depthBufferSize,
            remoteCamera, quadBuffers, colorTexture,
            mesh, false);
}

void MeshFromQuads::appendGeometry(
        unsigned int numProxies, const glm::uvec2 &depthBufferSize,
        const PerspectiveCamera &remoteCamera,
        const QuadBuffers &quadBuffers,
        const Texture &depthOffsetsBuffer,
        const Texture &colorTexture,
        const Mesh &mesh) {
    createMeshFromQuadsShader.bind();
    createMeshFromQuadsShader.setImageTexture(0, depthOffsetsBuffer, 0, GL_FALSE, 0, GL_READ_ONLY, depthOffsetsBuffer.internalFormat);

    createMeshFromProxies(
            numProxies, depthBufferSize,
            remoteCamera, quadBuffers, colorTexture,
            mesh, true);
}

void MeshFromQuads::createMeshFromProxies(
        unsigned int numProxies, const glm::uvec2 &depthBufferSize,
        const PerspectiveCamera &remoteCamera,
        const QuadBuffers &quadBuffers,
        const Texture &colorTexture,
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
        createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, sizesBuffer);
        createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, atlasOffsetBuffer);

        createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 2, mesh.vertexBuffer);
        createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 3, mesh.indexBuffer);
        createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 4, mesh.indirectBuffer);

        createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 5, quadBuffers.normalSphericalsBuffer);
        createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 6, quadBuffers.depthsBuffer);
        createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 7, quadBuffers.xysBuffer);
        createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 8, quadBuffers.offsetSizeFlattenedsBuffer);

        // createMeshFromQuadsShader.setImageTexture(1, colorTexture, 0, GL_FALSE, 0, GL_READ_ONLY, colorTexture.internalFormat);
        // createMeshFromQuadsShader.setImageTexture(2, atlas, 0, GL_FALSE, 0, GL_WRITE_ONLY, atlas.internalFormat);
    }
    createMeshFromQuadsShader.dispatch((numProxies + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1, 1);
    createMeshFromQuadsShader.memoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);

    createMeshFromQuadsShader.endTiming();
    stats.timeToCreateMeshMs = createMeshFromQuadsShader.getElapsedTime();
}
