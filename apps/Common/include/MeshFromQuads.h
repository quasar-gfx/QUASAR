#ifndef MESH_FROM_QUADS_H
#define MESH_FROM_QUADS_H

#include <Cameras/PerspectiveCamera.h>
#include <Primitives/Mesh.h>
#include <Shaders/ComputeShader.h>
#include <QuadsGenerator.h>
#include <Utils/TimeUtils.h>

#define THREADS_PER_LOCALGROUP 16

#define VERTICES_IN_A_QUAD 4
#define NUM_SUB_QUADS 4

class MeshFromQuads {
public:
    struct BufferSizes {
        unsigned int numVertices;
        unsigned int numIndices;
    };

    glm::uvec2 remoteWindowSize;

    struct Stats {
        double timeToCreateMeshMs = -1.0f;
    } stats;

    MeshFromQuads(const glm::uvec2 &remoteWindowSize)
            : remoteWindowSize(remoteWindowSize)
            , sizesBuffer(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_COPY, 1, &bufferSizes)
            , createMeshFromQuadsShader({
                .computeCodePath = "shaders/createMeshFromQuads.comp",
                .defines = {
                    "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
                }
            }) {
        createMeshFromQuadsShader.bind();
        createMeshFromQuadsShader.setVec2("remoteWindowSize", remoteWindowSize);
    }

    void createMeshFromProxies(
                unsigned int numProxies, const glm::uvec2 &depthBufferSize,
                const PerspectiveCamera &remoteCamera,
                const Buffer<unsigned int> &outputNormalSphericalsBuffer,
                const Buffer<float> &outputDepthsBuffer,
                const Buffer<unsigned int> &outputXYsBuffer,
                const Buffer<unsigned int> &outputOffsetSizeFlattenedsBuffer,
                const Texture& depthOffsetsBuffer,
                const Mesh& mesh
            ) {
        double startTime = timeutils::getTimeMicros();

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
            createMeshFromQuadsShader.setInt("quadMapSize", numProxies);
            createMeshFromQuadsShader.setVec2("depthBufferSize", depthBufferSize);
        }
        {
            createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, sizesBuffer);

            createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, mesh.vertexBuffer);
            createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 2, mesh.indexBuffer);
            createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 3, mesh.indirectBuffer);

            createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 4, outputNormalSphericalsBuffer);
            createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 5, outputDepthsBuffer);
            createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 6, outputXYsBuffer);
            createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 7, outputOffsetSizeFlattenedsBuffer);

            createMeshFromQuadsShader.setImageTexture(0, depthOffsetsBuffer, 0, GL_FALSE, 0, GL_READ_ONLY, depthOffsetsBuffer.internalFormat);
        }
        createMeshFromQuadsShader.dispatch((numProxies + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1, 1);
        createMeshFromQuadsShader.memoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);

        stats.timeToCreateMeshMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
    }

    void createMeshFromProxies(
                unsigned int numProxies, const glm::uvec2 &depthBufferSize,
                const PerspectiveCamera &remoteCamera,
                const Buffer<unsigned int> &outputNormalSphericalsBuffer,
                const Buffer<float> &outputDepthsBuffer,
                const Buffer<unsigned int> &outputXYsBuffer,
                const Buffer<unsigned int> &outputOffsetSizeFlattenedsBuffer,
                const Mesh& mesh
            ) {
        double startTime = timeutils::getTimeMicros();

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
            createMeshFromQuadsShader.setInt("quadMapSize", numProxies);
            createMeshFromQuadsShader.setVec2("depthBufferSize", depthBufferSize);
        }
        {
            createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, sizesBuffer);

            createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, mesh.vertexBuffer);
            createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 2, mesh.indexBuffer);
            createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 3, mesh.indirectBuffer);

            createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 4, outputNormalSphericalsBuffer);
            createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 5, outputDepthsBuffer);
            createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 6, outputXYsBuffer);
            createMeshFromQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 7, outputOffsetSizeFlattenedsBuffer);

            // createMeshFromQuadsShader.setImageTexture(0, depthOffsetsBuffer, 0, GL_FALSE, 0, GL_READ_ONLY, depthOffsetsBuffer.internalFormat);
        }
        createMeshFromQuadsShader.dispatch((numProxies + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1, 1);
        createMeshFromQuadsShader.memoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);

        stats.timeToCreateMeshMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
    }

    BufferSizes getBufferSizes() {
        sizesBuffer.bind();
        sizesBuffer.getData(&bufferSizes);
        return bufferSizes;
    }

private:
    BufferSizes bufferSizes = { 0 };
    Buffer<BufferSizes> sizesBuffer;

    ComputeShader createMeshFromQuadsShader;
};

#endif // MESH_FROM_QUADS_H
