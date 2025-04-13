#ifndef EDP_SIMULATOR_H
#define EDP_SIMULATOR_H

#include <PostProcessing/ShowNormalsEffect.h>

#include <Quads/QuadsGenerator.h>
#include <Quads/MeshFromQuads.h>
#include <Quads/QuadMaterial.h>
#include <Quads/FrameGenerator.h>

namespace quasar {

class DPSimulator {
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
    MeshFromQuads& meshFromQuads;
    FrameGenerator& frameGenerator;

    // reference frame (key frame)
    FrameRenderTarget refFrameRT;
    std::vector<Mesh> refFrameMeshes;
    std::vector<Node> refFrameNodes;
    std::vector<Node> refFrameWireframesLocal;

    // mask frame (residual frame) -- we only apply the mask to the visible layer
    FrameRenderTarget maskFrameRT;
    FrameRenderTarget maskTempRT;
    Mesh maskFrameMesh;
    Node maskFrameNode;

    // local objects
    std::vector<Node> refFrameNodesLocal;
    Node maskFrameWireframeNodesLocal;

    // hidden layers
    std::vector<FrameRenderTarget> frameRTsHidLayer;
    std::vector<Mesh> meshesHidLayer;
    std::vector<Node> nodesHidLayer;
    std::vector<Node> wireframesHidLayer;

    std::vector<Mesh> depthMeshsHidLayer;
    std::vector<Node> depthNodesHidLayer;

    // wide fov
    FrameRenderTarget wideFovRT;
    std::vector<Node> wideFovNode;

    Mesh depthMesh;
    Node depthNode;

    unsigned int maxLayers;

    int currMeshIndex = 0, prevMeshIndex = 1;

    unsigned int maxVertices = MAX_NUM_PROXIES * NUM_SUB_QUADS * VERTICES_IN_A_QUAD;
    unsigned int maxIndices = MAX_NUM_PROXIES * NUM_SUB_QUADS * INDICES_IN_A_QUAD;
    unsigned int maxVerticesDepth;

    std::vector<std::vector<char>> quads;
    std::vector<std::vector<char>> depthOffsets;

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

    DPSimulator(
                const PerspectiveCamera &remoteCamera,
                unsigned int maxLayers,
                QuadsGenerator &quadsGenerator, MeshFromQuads &meshFromQuads, FrameGenerator &frameGenerator)
            : quadsGenerator(quadsGenerator)
            , meshFromQuads(meshFromQuads)
            , frameGenerator(frameGenerator)
            , maxLayers(maxLayers)
            , quads(maxLayers)
            , depthOffsets(maxLayers)
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
            , wideFovRT({
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
        wideFovNode.reserve(2);
        refFrameNodesLocal.reserve(2);
        refFrameWireframesLocal.reserve(2);

        unsigned int numHidLayers = maxLayers - 1;

        frameRTsHidLayer.reserve(numHidLayers);
        meshesHidLayer.reserve(numHidLayers);
        depthMeshsHidLayer.reserve(numHidLayers);
        nodesHidLayer.reserve(numHidLayers);
        wireframesHidLayer.reserve(numHidLayers);
        depthNodesHidLayer.reserve(numHidLayers);

        // setup visible layer for reference frame
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

        // setup masks for residual frame
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

        // setup depth mesh
        MeshSizeCreateParams depthMeshParams = {
            .maxVertices = maxVerticesDepth,
            .usage = GL_DYNAMIC_DRAW
        };
        depthMesh = Mesh(depthMeshParams);
        depthNode.setEntity(&depthMesh);
        depthNode.frustumCulled = false;
        depthNode.visible = false;
        depthNode.primativeType = GL_POINTS;

        // setup hidden layers and wide fov RTs
        RenderTargetCreateParams rtParams = {
            .width = quadsGenerator.remoteWindowSize.x,
            .height = quadsGenerator.remoteWindowSize.y,
            .internalFormat = GL_RGBA16F,
            .format = GL_RGBA,
            .type = GL_HALF_FLOAT,
            .wrapS = GL_CLAMP_TO_EDGE,
            .wrapT = GL_CLAMP_TO_EDGE,
            .minFilter = GL_NEAREST,
            .magFilter = GL_NEAREST
        };
        for (int layer = 0; layer < numHidLayers; layer++) {
            if (layer == numHidLayers - 1) {
                rtParams.width /= 2; rtParams.height /= 2;
            }
            frameRTsHidLayer.emplace_back(rtParams);
        }

        for (int view = 0; view < numHidLayers; view++) {
            if (view == numHidLayers - 1) {
                meshParams.material = new QuadMaterial({ .baseColorTexture = &wideFovRT.colorBuffer });
            }
            else {
                meshParams.material = new QuadMaterial({ .baseColorTexture = &frameRTsHidLayer[view].colorBuffer });
            }
            // we can use less vertices and indicies for the hidden layers since they will be sparse
            meshParams.maxVertices = maxVertices / 4;
            meshParams.maxIndices = maxIndices / 4;
            meshesHidLayer.emplace_back(meshParams);

            nodesHidLayer.emplace_back(&meshesHidLayer[view]);
            nodesHidLayer[view].frustumCulled = false;

            const glm::vec4 &color = colors[(view + 1) % colors.size()];

            wireframesHidLayer.emplace_back(&meshesHidLayer[view]);
            wireframesHidLayer[view].frustumCulled = false;
            wireframesHidLayer[view].wireframe = true;
            wireframesHidLayer[view].overrideMaterial = new QuadMaterial({ .baseColor = color });

            depthMeshParams.material = new UnlitMaterial({ .baseColor = color });
            depthMeshsHidLayer.emplace_back(depthMeshParams);

            depthNodesHidLayer.emplace_back(&depthMeshsHidLayer[view]);
            depthNodesHidLayer[view].frustumCulled = false;
            depthNodesHidLayer[view].primativeType = GL_POINTS;
        }

        // setup scene to use as mask for wide fov camera
        for (int i = 0; i < 2; i++) {
            wideFovNode.emplace_back(&refFrameMeshes[i]);
            wideFovNode[i].frustumCulled = false;
            sceneWideFov.addChildNode(&wideFovNode[i]);
        }
        for (int i = 0; i < numHidLayers - 1; i++) {
            sceneWideFov.addChildNode(&nodesHidLayer[i]);
        }
        sceneWideFov.addChildNode(&maskFrameNode);
    }
    ~DPSimulator() = default;

    void addMeshesToScene(Scene& localScene) {
        for (int i = 0; i < 2; i++) {
            localScene.addChildNode(&refFrameNodesLocal[i]);
            localScene.addChildNode(&refFrameWireframesLocal[i]);
        }
        localScene.addChildNode(&maskFrameNode);
        localScene.addChildNode(&maskFrameWireframeNodesLocal);
        localScene.addChildNode(&depthNode);

        for (int view = 0; view < maxLayers - 1; view++) {
            localScene.addChildNode(&nodesHidLayer[view]);
            localScene.addChildNode(&wireframesHidLayer[view]);
            localScene.addChildNode(&depthNodesHidLayer[view]);
        }
    }

    void generateFrame(
            const PerspectiveCamera& remoteCameraCenter, const PerspectiveCamera& remoteCameraWideFov, const Scene& remoteScene,
            DeferredRenderer& remoteRenderer,
            DepthPeelingRenderer& remoteRendererDP,
            bool generateMaskFrame = false, bool showNormals = false, bool showDepth = false) {
        // reset stats
        stats = { 0 };

        double startTime = timeutils::getTimeMicros();
        // render remote scene with multiple layers
        remoteRendererDP.drawObjects(remoteScene, remoteCameraCenter);
        stats.totalRenderTime += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        for (int view = 0; view < maxLayers; view++) {
            int hiddenIndex = view - 1;
            auto& remoteCameraToUse = (view == 0 && generateMaskFrame) ? remoteCameraPrev :
                                        ((view != maxLayers - 1) ? remoteCameraCenter : remoteCameraWideFov);

            auto& frameToUse = (view == 0) ? refFrameRT : frameRTsHidLayer[hiddenIndex];

            auto& meshToUse = (view == 0) ? refFrameMeshes[currMeshIndex] : meshesHidLayer[hiddenIndex];
            auto& meshToUseDepth = (view == 0) ? depthMesh : depthMeshsHidLayer[hiddenIndex];

            startTime = timeutils::getTimeMicros();
            if (view == 0) {
                remoteRenderer.drawObjectsNoLighting(remoteScene, remoteCameraToUse);
                if (!showNormals) {
                    remoteRenderer.copyToFrameRT(frameToUse);
                }
                else {
                    showNormalsEffect.drawToRenderTarget(remoteRenderer, frameToUse);
                }
            }
            else if (view != maxLayers - 1) {
                // copy to render target
                if (!showNormals) {
                    remoteRendererDP.peelingLayers[hiddenIndex+1].blitToFrameRT(frameToUse);
                }
                else {
                    showNormalsEffect.drawToRenderTarget(remoteRendererDP, frameToUse);
                }
            }
            // wide fov camera
            else {
                // draw old center mesh at new remoteCamera view, filling stencil buffer with 1
                remoteRenderer.pipeline.stencilState.enableRenderingIntoStencilBuffer(GL_KEEP, GL_KEEP, GL_REPLACE);
                remoteRenderer.pipeline.writeMaskState.disableColorWrites();
                wideFovNode[currMeshIndex].visible = false;
                wideFovNode[prevMeshIndex].visible = true;
                remoteRenderer.drawObjectsNoLighting(sceneWideFov, remoteCameraToUse);

                // render remoteScene using stencil buffer as a mask
                // at values where stencil buffer is not 1, remoteScene should render
                remoteRenderer.pipeline.stencilState.enableRenderingUsingStencilBufferAsMask(GL_NOTEQUAL, 1);
                remoteRenderer.pipeline.writeMaskState.enableColorWrites();
                remoteRenderer.drawObjectsNoLighting(remoteScene, remoteCameraToUse, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                remoteRenderer.pipeline.stencilState.restoreStencilState();

                if (!showNormals) {
                    remoteRenderer.copyToFrameRT(frameToUse);
                    remoteRenderer.copyToFrameRT(wideFovRT);
                }
                else {
                    showNormalsEffect.drawToRenderTarget(remoteRenderer, frameToUse);
                    showNormalsEffect.drawToRenderTarget(remoteRenderer, wideFovRT);
                }
            }
            stats.totalRenderTime += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

            /*
            ============================
            Generate I-frame
            ============================
            */
            unsigned int numProxies = 0, numDepthOffsets = 0;
            quadsGenerator.expandEdges = false;
            if (view != 0) {
                quadsGenerator.depthThreshold *= 10.0f;
                quadsGenerator.flattenThreshold *= 10.0f;
                quadsGenerator.proxySimilarityThreshold *= 10.0f;
            }
            unsigned int numBytesIFrame = frameGenerator.generateRefFrame(
                frameToUse, remoteCameraToUse,
                meshToUse,
                quads[view], depthOffsets[view],
                numProxies, numDepthOffsets
            );
            if (!generateMaskFrame) {
                stats.compressedSizeBytes += numBytesIFrame;
            }
            if (view != 0) {
                quadsGenerator.depthThreshold /= 10.0f;
                quadsGenerator.flattenThreshold /= 10.0f;
                quadsGenerator.proxySimilarityThreshold /= 10.0f;
            }

            stats.totalGenQuadMapTime += frameGenerator.stats.timeToGenerateQuadsMs;
            stats.totalSimplifyTime += frameGenerator.stats.timeToSimplifyQuadsMs;
            stats.totalFillQuadsTime += frameGenerator.stats.timeToFillOutputQuadsMs;
            stats.totalCreateProxiesTime += frameGenerator.stats.timeToCreateProxiesMs;

            stats.totalAppendProxiesTime += frameGenerator.stats.timeToAppendProxiesMs;
            stats.totalFillQuadsIndiciesTime += frameGenerator.stats.timeToFillQuadIndicesMs;
            stats.totalCreateVertIndTime += frameGenerator.stats.timeToCreateVertIndMs;
            stats.totalCreateMeshTime += frameGenerator.stats.timeToCreateMeshMs;

            if (view != 0 || !generateMaskFrame) {
                stats.totalCompressTime += frameGenerator.stats.timeToCompress;
            }

            /*
            ============================
            Generate P-frame
            ============================
            */
            if (view == 0) {
                if (generateMaskFrame) {
                    quadsGenerator.expandEdges = true;
                    stats.compressedSizeBytes += frameGenerator.generateMaskFrame(
                        meshScenes[currMeshIndex], meshScenes[prevMeshIndex],
                        maskTempRT, maskFrameRT,
                        remoteCameraCenter, remoteCameraPrev,
                        refFrameMeshes[currMeshIndex], maskFrameMesh,
                        quads[view], depthOffsets[view],
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
                    remoteCameraPrev.setViewMatrix(remoteCameraCenter.getViewMatrix());
                }
            }

            // for debugging: Generate point cloud from depth map
            if (showDepth) {
                const glm::vec2 frameSize = glm::vec2(frameToUse.width, frameToUse.height);

                startTime = timeutils::getTimeMicros();

                meshFromDepthShader.bind();
                {
                    meshFromDepthShader.setTexture(frameToUse.depthStencilBuffer, 0);
                }
                {
                    meshFromDepthShader.setVec2("depthMapSize", frameSize);
                }
                {
                    meshFromDepthShader.setMat4("view", remoteCameraToUse.getViewMatrix());
                    meshFromDepthShader.setMat4("projection", remoteCameraToUse.getProjectionMatrix());
                    meshFromDepthShader.setMat4("viewInverse", remoteCameraToUse.getViewMatrixInverse());
                    meshFromDepthShader.setMat4("projectionInverse", remoteCameraToUse.getProjectionMatrixInverse());

                    meshFromDepthShader.setFloat("near", remoteCameraToUse.getNear());
                    meshFromDepthShader.setFloat("far", remoteCameraToUse.getFar());
                }
                {
                    meshFromDepthShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, meshToUseDepth.vertexBuffer);
                    meshFromDepthShader.clearBuffer(GL_SHADER_STORAGE_BUFFER, 1);
                }
                meshFromDepthShader.dispatch((frameSize.x + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                             (frameSize.y + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
                meshFromDepthShader.memoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);

                stats.totalGenDepthTime += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
            }
        }
    }

    unsigned int saveToFile(const std::string &outputPath) {
        unsigned int totalOutputSize = 0;
        for (int view = 0; view < maxLayers; view++) {
            // save quads
            double startTime = timeutils::getTimeMicros();
            std::string filename = outputPath + "quads" + std::to_string(view) + ".bin";
            std::ofstream quadsFile = std::ofstream(filename + ".zstd", std::ios::binary);
            quadsFile.write(quads[view].data(), quads[view].size());
            quadsFile.close();
            spdlog::info("Saved {} quads ({:.3f}MB) in {:.3f}ms",
                        stats.totalProxies, static_cast<double>(quads[view].size()) / BYTES_IN_MB,
                            timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

            // save depth offsets
            startTime = timeutils::getTimeMicros();
            std::string depthOffsetsFileName = outputPath + "depthOffsets" + std::to_string(view) + ".bin";
            std::ofstream depthOffsetsFile = std::ofstream(depthOffsetsFileName + ".zstd", std::ios::binary);
            depthOffsetsFile.write(depthOffsets[view].data(), depthOffsets[view].size());
            depthOffsetsFile.close();
            spdlog::info("Saved {} depth offsets ({:.3f}MB) in {:.3f}ms",
                        stats.totalDepthOffsets, static_cast<double>(depthOffsets[view].size()) / BYTES_IN_MB,
                            timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

            // save color buffer
            std::string colorFileName = outputPath + "color" + std::to_string(view) + ".png";
            auto& frameRTToUse = (view == 0) ? refFrameRT : frameRTsHidLayer[view-1];
            frameRTToUse.saveColorAsPNG(colorFileName);

            totalOutputSize += quads[view].size() + depthOffsets[view].size();
        }

        return totalOutputSize;
    }

private:
    // shaders
    ShowNormalsEffect showNormalsEffect;
    ComputeShader meshFromDepthShader;

    PerspectiveCamera remoteCameraPrev;

    // scenes with resulting mesh
    std::vector<Scene> meshScenes;
    Scene sceneWideFov;

    QuadMaterial wireframeMaterial = QuadMaterial({.baseColor = colors[0]});
    QuadMaterial maskWireframeMaterial = QuadMaterial({.baseColor = colors[colors.size()-1]});
    UnlitMaterial depthMaterial = UnlitMaterial({.baseColor = colors[2]});
};

} // namespace quasar


#endif // EDP_SIMULATOR_H
