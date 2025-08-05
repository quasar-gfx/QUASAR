#ifndef DEPTH_MESH_H
#define DEPTH_MESH_H

#include <Primitives/Mesh.h>
#include <Shaders/ComputeShader.h>
#include <RenderTargets/FrameRenderTarget.h>

#include <shaders_common.h>

#ifndef __ANDROID__
#define THREADS_PER_LOCALGROUP 16
#else
#define THREADS_PER_LOCALGROUP 32
#endif

namespace quasar {

class DepthMesh : public Mesh {
public:
    const glm::uvec2& depthMapSize;

    struct Stats {
        double genDepthTime = 0.0;
    } stats;

    DepthMesh(const glm::uvec2& depthMapSize, const glm::vec4& color = glm::vec4(1.0f))
        : depthMapSize(depthMapSize)
        , meshFromDepthShader({
            .computeCodeData = SHADER_COMMON_MESH_FROM_DEPTH_COMP,
            .computeCodeSize = SHADER_COMMON_MESH_FROM_DEPTH_COMP_len,
            .defines = {
                "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
            }
        })
        , depthMaterial({ .baseColor = color })
        , Mesh({
            .maxVertices = depthMapSize.x * depthMapSize.y,
            .maxIndices = 0,
            .material = &depthMaterial,
            .usage = GL_DYNAMIC_DRAW,
        })
    {}

    void update(const PerspectiveCamera& camera, const FrameRenderTarget& rt) {
        meshFromDepthShader.startTiming();

        meshFromDepthShader.bind();
        {
            meshFromDepthShader.setTexture(rt.depthStencilBuffer, 0);
        }
        {
            meshFromDepthShader.setVec2("depthMapSize", depthMapSize);
        }
        {
            meshFromDepthShader.setMat4("view", camera.getViewMatrix());
            meshFromDepthShader.setMat4("projection", camera.getProjectionMatrix());
            meshFromDepthShader.setMat4("viewInverse", camera.getViewMatrixInverse());
            meshFromDepthShader.setMat4("projectionInverse", camera.getProjectionMatrixInverse());

            meshFromDepthShader.setFloat("near", camera.getNear());
            meshFromDepthShader.setFloat("far", camera.getFar());
        }
        {
            meshFromDepthShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, vertexBuffer);
            meshFromDepthShader.clearBuffer(GL_SHADER_STORAGE_BUFFER, 1);
        }
        meshFromDepthShader.dispatch((depthMapSize.x + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                     (depthMapSize.y + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
        meshFromDepthShader.memoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);

        meshFromDepthShader.endTiming();
        stats.genDepthTime += meshFromDepthShader.getElapsedTime();
    }

private:
    ComputeShader meshFromDepthShader;
    UnlitMaterial depthMaterial;
};

} // namespace quasar

#endif // DEPTH_MESH_H
