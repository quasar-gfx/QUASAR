#ifndef QUADS_SIMULATOR_H
#define QUADS_SIMULATOR_H

#include <PostProcessing/ToneMapper.h>
#include <PostProcessing/ShowNormalsEffect.h>

#include <Quads/QuadsGenerator.h>
#include <Quads/MeshFromQuads.h>
#include <Quads/QuadMaterial.h>
#include <Quads/FrameGenerator.h>

#include <shaders_common.h>

namespace quasar {

class QuadsSimulator {
public:
    const std::vector<glm::vec4> colors = {
        glm::vec4(1.0f, 1.0f, 0.0f, 1.0f), // primary view color is yellow
        glm::vec4(0.0f, 0.0f, 1.0f, 1.0f),
        glm::vec4(0.0f, 1.0f, 0.0f, 1.0f),
        glm::vec4(1.0f, 0.5f, 0.5f, 1.0f),
        glm::vec4(0.5f, 0.0f, 0.0f, 1.0f),
        glm::vec4(0.0f, 1.0f, 1.0f, 1.0f),
        glm::vec4(1.0f, 0.0f, 0.0f, 1.0f),
        glm::vec4(0.0f, 0.5f, 0.0f, 1.0f),
        glm::vec4(0.0f, 0.0f, 0.5f, 1.0f),
        glm::vec4(0.5f, 0.0f, 0.5f, 1.0f),
    };

    QuadsGenerator& quadsGenerator;
    FrameGenerator& frameGenerator;

    // reference frame (key frame)
    FrameRenderTarget refFrameRT;
    std::vector<Mesh> refFrameMeshes;
    std::vector<Node> refFrameNodes;

    // mask frame (residual frame)
    FrameRenderTarget maskFrameRT;
    FrameRenderTarget maskTempRT;
    Mesh maskFrameMesh;
    Node maskFrameNode;

    // local objects
    std::vector<Node> refFrameNodesLocal;
    std::vector<Node> refFrameWireframesLocal;
    Node maskFrameWireframeNodesLocal;

    Mesh depthMesh;
    Node depthNode;

    FrameRenderTarget copyRT;

    int currMeshIndex = 0, prevMeshIndex = 1;

    unsigned int maxVertices = MAX_NUM_PROXIES * NUM_SUB_QUADS * VERTICES_IN_A_QUAD;
    unsigned int maxIndices = MAX_NUM_PROXIES * NUM_SUB_QUADS * INDICES_IN_A_QUAD;
    unsigned int maxVerticesDepth;

    std::vector<char> quads;
    std::vector<char> depthOffsets;

    struct Stats {
        double totalRenderTime = 0.0;
        double totalCreateProxiesTime = 0.0;
        double totalGenQuadMapTime = 0.0;
        double totalSimplifyTime = 0.0;
        double totalFillQuadsTime = 0.0;
        double totalCreateMeshTime = 0.0;
        double totalAppendProxiesTime = 0.0;
        double totalFillQuadsIndiciesTime = 0.0;
        double totalCreateVertIndTime = 0.0;
        double totalGenDepthTime = 0.0;
        double totalCompressTime = 0.0;

        unsigned int totalProxies = 0;
        unsigned int totalDepthOffsets = 0;
        double compressedSizeBytes = 0;
    } stats;

    QuadsSimulator(const PerspectiveCamera &remoteCamera, FrameGenerator &frameGenerator)
            : quadsGenerator(frameGenerator.quadsGenerator)
            , frameGenerator(frameGenerator)
            , maxVerticesDepth(quadsGenerator.remoteWindowSize.x * quadsGenerator.remoteWindowSize.y)
            , meshFromDepthShader({
                .computeCodeData = SHADER_COMMON_MESHFROMDEPTH_COMP,
                .computeCodeSize = SHADER_COMMON_MESHFROMDEPTH_COMP_len,
                .defines = {
                    "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
                }
            })
            , refFrameRT({
                .width = quadsGenerator.remoteWindowSize.x,
                .height = quadsGenerator.remoteWindowSize.y,
                .internalFormat = GL_RGBA16F,
                .format = GL_RGBA,
                .type = GL_HALF_FLOAT,
                .wrapS = GL_CLAMP_TO_EDGE,
                .wrapT = GL_CLAMP_TO_EDGE,
                .minFilter = GL_NEAREST,
                .magFilter = GL_NEAREST
            })
            , maskFrameRT({
                .width = quadsGenerator.remoteWindowSize.x,
                .height = quadsGenerator.remoteWindowSize.y,
                .internalFormat = GL_RGBA16F,
                .format = GL_RGBA,
                .type = GL_HALF_FLOAT,
                .wrapS = GL_CLAMP_TO_EDGE,
                .wrapT = GL_CLAMP_TO_EDGE,
                .minFilter = GL_NEAREST,
                .magFilter = GL_NEAREST
            })
            , maskTempRT({
                .width = quadsGenerator.remoteWindowSize.x,
                .height = quadsGenerator.remoteWindowSize.y,
                .internalFormat = GL_RGBA16F,
                .format = GL_RGBA,
                .type = GL_HALF_FLOAT,
                .wrapS = GL_CLAMP_TO_EDGE,
                .wrapT = GL_CLAMP_TO_EDGE,
                .minFilter = GL_NEAREST,
                .magFilter = GL_NEAREST
            })
            , copyRT({
                .width = quadsGenerator.remoteWindowSize.x,
                .height = quadsGenerator.remoteWindowSize.y,
                .internalFormat = GL_RGBA16F,
                .format = GL_RGBA,
                .type = GL_HALF_FLOAT,
                .wrapS = GL_CLAMP_TO_EDGE,
                .wrapT = GL_CLAMP_TO_EDGE,
                .minFilter = GL_NEAREST,
                .magFilter = GL_NEAREST
            })
            , meshScenes(2) {
        remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());
        remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());

        refFrameMeshes.reserve(2);
        refFrameNodes.reserve(2);
        refFrameNodesLocal.reserve(2);
        refFrameWireframesLocal.reserve(2);

        MeshSizeCreateParams meshParams({
            .maxVertices = maxVertices,
            .maxIndices = maxIndices,
            .vertexSize = sizeof(QuadVertex),
            .attributes = QuadVertex::getVertexInputAttributes(),
            .material = new QuadMaterial({ .baseColorTexture = &refFrameRT.colorBuffer }),
            .usage = GL_DYNAMIC_DRAW,
            .indirectDraw = true
        });
        for (int i = 0; i < 2; i++) {
            refFrameMeshes.emplace_back(meshParams);

            refFrameNodes.emplace_back(&refFrameMeshes[i]);
            refFrameNodes[i].frustumCulled = false;
            meshScenes[i].addChildNode(&refFrameNodes[i]);

            refFrameNodesLocal.emplace_back(&refFrameMeshes[i]);
            refFrameNodesLocal[i].frustumCulled = false;

            refFrameWireframesLocal.emplace_back(&refFrameMeshes[i]);
            refFrameWireframesLocal[i].frustumCulled = false;
            refFrameWireframesLocal[i].wireframe = true;
            refFrameWireframesLocal[i].visible = false;
            refFrameWireframesLocal[i].overrideMaterial = &wireframeMaterial;
        }

        // we can use less vertices and indicies for the mask since it will be sparse
        meshParams.maxVertices /= 4;
        meshParams.maxIndices /= 4;
        meshParams.material = new QuadMaterial({ .baseColorTexture = &maskFrameRT.colorBuffer });
        maskFrameMesh = Mesh(meshParams);
        maskFrameNode.setEntity(&maskFrameMesh);
        maskFrameNode.frustumCulled = false;

        maskFrameWireframeNodesLocal.setEntity(&maskFrameMesh);
        maskFrameWireframeNodesLocal.frustumCulled = false;
        maskFrameWireframeNodesLocal.wireframe = true;
        maskFrameWireframeNodesLocal.visible = false;
        maskFrameWireframeNodesLocal.overrideMaterial = &maskWireframeMaterial;

        depthMesh = Mesh({
            .maxVertices = maxVerticesDepth,
            .material = &depthMaterial,
            .usage = GL_DYNAMIC_DRAW
        });
        depthNode.setEntity(&depthMesh);
        depthNode.frustumCulled = false;
        depthNode.visible = false;
        depthNode.primativeType = GL_POINTS;
    }
    ~QuadsSimulator() = default;

    void addMeshesToScene(Scene& localScene) {
        for (int i = 0; i < 2; i++) {
            localScene.addChildNode(&refFrameNodesLocal[i]);
            localScene.addChildNode(&refFrameWireframesLocal[i]);
        }
        localScene.addChildNode(&maskFrameNode);
        localScene.addChildNode(&maskFrameWireframeNodesLocal);
        localScene.addChildNode(&depthNode);
    }

    void generateFrame(
            const PerspectiveCamera& remoteCamera, const Scene& remoteScene,
            DeferredRenderer& remoteRenderer,
            bool generateMaskFrame = false, bool showNormals = false, bool showDepth = false) {
        double startTime = timeutils::getTimeMicros();

        auto& remoteCameraToUse = generateMaskFrame ? remoteCameraPrev : remoteCamera;

        // reset stats
        stats = { 0 };

        // render all objects in remoteScene normally
        remoteRenderer.drawObjects(remoteScene, remoteCameraToUse);
        if (!showNormals) {
            remoteRenderer.copyToFrameRT(refFrameRT);
            toneMapper.drawToRenderTarget(remoteRenderer, copyRT);
        }
        else {
            showNormalsEffect.drawToRenderTarget(remoteRenderer, refFrameRT);
        }
        stats.totalRenderTime += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        /*
        ============================
        Generate I-frame
        ============================
        */
        unsigned int numProxies = 0, numDepthOffsets = 0;
        quadsGenerator.expandEdges = false;
        stats.compressedSizeBytes = frameGenerator.generateRefFrame(
            refFrameRT,
            remoteCameraToUse,
            refFrameMeshes[currMeshIndex],
            quads, depthOffsets,
            numProxies, numDepthOffsets
        );
        stats.totalGenQuadMapTime += frameGenerator.stats.timeToGenerateQuadsMs;
        stats.totalSimplifyTime += frameGenerator.stats.timeToSimplifyQuadsMs;
        stats.totalFillQuadsTime += frameGenerator.stats.timeToFillOutputQuadsMs;
        stats.totalCreateProxiesTime += frameGenerator.stats.timeToCreateProxiesMs;

        stats.totalAppendProxiesTime += frameGenerator.stats.timeToAppendProxiesMs;
        stats.totalFillQuadsIndiciesTime += frameGenerator.stats.timeToFillQuadIndicesMs;
        stats.totalCreateVertIndTime += frameGenerator.stats.timeToCreateVertIndMs;
        stats.totalCreateMeshTime += frameGenerator.stats.timeToCreateMeshMs;

        if (!generateMaskFrame) {
            stats.totalCompressTime += frameGenerator.stats.timeToCompress;
        }

        /*
        ============================
        Generate P-frame
        ============================
        */
        if (generateMaskFrame) {
            quadsGenerator.expandEdges = true;
            stats.compressedSizeBytes = frameGenerator.generateMaskFrame(
                meshScenes[currMeshIndex], meshScenes[prevMeshIndex],
                maskTempRT, maskFrameRT,
                remoteCamera, remoteCameraPrev,
                refFrameMeshes[currMeshIndex], maskFrameMesh,
                quads, depthOffsets,
                numProxies, numDepthOffsets
            );

            stats.totalRenderTime += frameGenerator.stats.timeToRenderMaskMs;

            stats.totalGenQuadMapTime += frameGenerator.stats.timeToGenerateQuadsMs;
            stats.totalSimplifyTime += frameGenerator.stats.timeToSimplifyQuadsMs;
            stats.totalFillQuadsTime += frameGenerator.stats.timeToFillOutputQuadsMs;
            stats.totalCreateProxiesTime += frameGenerator.stats.timeToCreateProxiesMs;

            stats.totalAppendProxiesTime += frameGenerator.stats.timeToAppendProxiesMs;
            stats.totalFillQuadsIndiciesTime += frameGenerator.stats.timeToFillOutputQuadsMs;
            stats.totalCreateVertIndTime += frameGenerator.stats.timeToCreateVertIndMs;
            stats.totalCreateMeshTime += frameGenerator.stats.timeToCreateMeshMs;

            stats.totalCompressTime += frameGenerator.stats.timeToCompress;
        }
        stats.totalProxies += numProxies;
        stats.totalDepthOffsets += numDepthOffsets;

        maskFrameNode.visible = generateMaskFrame;
        currMeshIndex = (currMeshIndex + 1) % 2;
        prevMeshIndex = (prevMeshIndex + 1) % 2;

        // only update the previous camera pose if we are not generating a P-Frame
        if (!generateMaskFrame) {
            remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());
        }

        // for debugging: Generate point cloud from depth map
        if (showDepth) {
            const glm::vec2 frameSize = glm::vec2(refFrameRT.width, refFrameRT.height);

            meshFromDepthShader.startTiming();

            meshFromDepthShader.bind();
            {
                meshFromDepthShader.setTexture(refFrameRT.depthStencilBuffer, 0);
            }
            {
                meshFromDepthShader.setVec2("depthMapSize", frameSize);
            }
            {
                meshFromDepthShader.setMat4("view", remoteCamera.getViewMatrix());
                meshFromDepthShader.setMat4("projection", remoteCamera.getProjectionMatrix());
                meshFromDepthShader.setMat4("viewInverse", remoteCamera.getViewMatrixInverse());
                meshFromDepthShader.setMat4("projectionInverse", remoteCamera.getProjectionMatrixInverse());

                meshFromDepthShader.setFloat("near", remoteCamera.getNear());
                meshFromDepthShader.setFloat("far", remoteCamera.getFar());
            }
            {
                meshFromDepthShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, depthMesh.vertexBuffer);
                meshFromDepthShader.clearBuffer(GL_SHADER_STORAGE_BUFFER, 1);
            }
            meshFromDepthShader.dispatch((frameSize.x + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                         (frameSize.y + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
            meshFromDepthShader.memoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);

            meshFromDepthShader.endTiming();
            stats.totalGenDepthTime += meshFromDepthShader.getElapsedTime();
        }
    }

    unsigned int saveToFile(const std::string &outputPath) {
        // save quads
        double startTime = timeutils::getTimeMicros();
        std::string filename = outputPath + "quads.bin";
        std::ofstream quadsFile = std::ofstream(filename + ".zstd", std::ios::binary);
        quadsFile.write(quads.data(), quads.size());
        quadsFile.close();
        spdlog::info("Saved {} quads ({:.3f}MB) in {:.3f}ms",
                      stats.totalProxies, static_cast<double>(quads.size()) / BYTES_IN_MB,
                        timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

        // save depth offsets
        startTime = timeutils::getTimeMicros();
        std::string depthOffsetsFileName = outputPath + "depthOffsets.bin";
        std::ofstream depthOffsetsFile = std::ofstream(depthOffsetsFileName + ".zstd", std::ios::binary);
        depthOffsetsFile.write(depthOffsets.data(), depthOffsets.size());
        depthOffsetsFile.close();
        spdlog::info("Saved {} depth offsets ({:.3f}MB) in {:.3f}ms",
                     stats.totalDepthOffsets, static_cast<double>(depthOffsets.size()) / BYTES_IN_MB,
                        timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

        // save color buffer
        std::string colorFileName = outputPath + "color.jpg";
        copyRT.saveColorAsJPG(colorFileName);

        return quads.size() + depthOffsets.size();
    }

private:
    // shaders
    ToneMapper toneMapper;
    ShowNormalsEffect showNormalsEffect;
    ComputeShader meshFromDepthShader;

    PerspectiveCamera remoteCameraPrev;

    // scenes with resulting mesh
    std::vector<Scene> meshScenes;

    QuadMaterial wireframeMaterial = QuadMaterial({.baseColor = colors[0]});
    QuadMaterial maskWireframeMaterial = QuadMaterial({.baseColor = colors[colors.size()-1]});
    UnlitMaterial depthMaterial = UnlitMaterial({.baseColor = colors[2]});
};

} // namespace quasar


#endif // QUADS_SIMULATOR_H
