#ifndef QUASAR_SIMULATOR_H
#define QUASAR_SIMULATOR_H

#include <PostProcessing/ToneMapper.h>
#include <PostProcessing/ShowNormalsEffect.h>

#include <DepthMesh.h>

#include <Quads/QuadMaterial.h>
#include <Quads/FrameGenerator.h>

namespace quasar {

class QRSimulator {
public:
    const std::vector<glm::vec4> colors = {
        glm::vec4(1.0f, 1.0f, 0.0f, 1.0f), // primary layer color is yellow
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

    // Reference frame
    FrameRenderTarget refFrameRT;
    std::vector<Mesh> refFrameMeshes;
    std::vector<Node> refFrameNodes;
    std::vector<Node> refFrameWireframesLocal;

    // Mask frame (residual frame) -- we only apply the mask to the visible layer
    FrameRenderTarget maskFrameRT;
    FrameRenderTarget maskTempRT;
    Mesh maskFrameMesh;
    Node maskFrameNode;

    // Local objects
    std::vector<Node> refFrameNodesLocal;
    Node maskFrameWireframeNodesLocal;

    // Hidden layers
    std::vector<FrameRenderTarget> frameRTsHidLayer;
    std::vector<Mesh> meshesHidLayer;
    std::vector<Node> nodesHidLayer;
    std::vector<Node> wireframesHidLayer;

    std::vector<DepthMesh> depthMeshsHidLayer;
    std::vector<Node> depthNodesHidLayer;

    // Wide fov
    std::vector<Node> wideFovNodes;

    std::vector<FrameRenderTarget> copyRTs;

    DepthMesh depthMesh;
    Node depthNode;

    uint maxLayers;

    int currMeshIndex = 0, prevMeshIndex = 1;

    uint maxVertices = MAX_NUM_PROXIES * NUM_SUB_QUADS * VERTICES_IN_A_QUAD;
    uint maxIndices = MAX_NUM_PROXIES * NUM_SUB_QUADS * INDICES_IN_A_QUAD;
    uint maxVerticesDepth;

    std::vector<std::vector<char>> quads;
    std::vector<std::vector<char>> depthOffsets;

    struct Stats {
        double totalRenderTime = 0.0;
        double totalCreateProxiesTime = 0.0;
        double totalGenQuadMapTime = 0.0;
        double totalSimplifyTime = 0.0;
        double totalGatherQuadsTime = 0.0;
        double totalCreateMeshTime = 0.0;
        double totalAppendQuadsTime = 0.0;
        double totalFillQuadsIndiciesTime = 0.0;
        double totalCreateVertIndTime = 0.0;
        double totalGenDepthTime = 0.0;
        double totalCompressTime = 0.0;

        uint totalProxies = 0;
        uint totalDepthOffsets = 0;
        double compressedSizeBytes = 0;
    } stats;

    QRSimulator(QuadFrame& quadFrame, uint maxLayers, const PerspectiveCamera& remoteCamera, FrameGenerator& frameGenerator)
        : quadFrame(quadFrame)
        , frameGenerator(frameGenerator)
        , maxLayers(maxLayers)
        , quads(maxLayers)
        , depthOffsets(maxLayers)
        , maxVerticesDepth(quadFrame.getSize().x * quadFrame.getSize().y)
        , refFrameRT({
            .width = quadFrame.getSize().x,
            .height = quadFrame.getSize().y,
            .internalFormat = GL_RGBA16F,
            .format = GL_RGBA,
            .type = GL_HALF_FLOAT,
            .wrapS = GL_CLAMP_TO_EDGE,
            .wrapT = GL_CLAMP_TO_EDGE,
            .minFilter = GL_NEAREST,
            .magFilter = GL_NEAREST,
        })
        , maskFrameRT({
            .width = quadFrame.getSize().x,
            .height = quadFrame.getSize().y,
            .internalFormat = GL_RGBA16F,
            .format = GL_RGBA,
            .type = GL_HALF_FLOAT,
            .wrapS = GL_CLAMP_TO_EDGE,
            .wrapT = GL_CLAMP_TO_EDGE,
            .minFilter = GL_NEAREST,
            .magFilter = GL_NEAREST,
        })
        , maskTempRT({
            .width = quadFrame.getSize().x,
            .height = quadFrame.getSize().y,
            .internalFormat = GL_RGBA16F,
            .format = GL_RGBA,
            .type = GL_HALF_FLOAT,
            .wrapS = GL_CLAMP_TO_EDGE,
            .wrapT = GL_CLAMP_TO_EDGE,
            .minFilter = GL_NEAREST,
            .magFilter = GL_NEAREST,
        })
        , depthMesh(quadFrame.getSize(), glm::vec4(0.0f, 1.0f, 0.0f, 1.0f))
        , meshScenes(2)
    {
        remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());
        remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());

        refFrameMeshes.reserve(2);
        refFrameNodes.reserve(2);
        wideFovNodes.reserve(2);
        refFrameNodesLocal.reserve(2);
        refFrameWireframesLocal.reserve(2);

        copyRTs.reserve(maxLayers);

        proxiesPerQuadSet.reserve(maxLayers);
        depthOffsetsPerQuadSet.reserve(maxLayers);

        uint numHidLayers = maxLayers - 1;

        frameRTsHidLayer.reserve(numHidLayers);
        meshesHidLayer.reserve(numHidLayers);
        depthMeshsHidLayer.reserve(numHidLayers);
        nodesHidLayer.reserve(numHidLayers);
        wireframesHidLayer.reserve(numHidLayers);
        depthNodesHidLayer.reserve(numHidLayers);

        // Setup visible layer for reference frame
        MeshSizeCreateParams meshParams({
            .maxVertices = maxVertices,
            .maxIndices = maxIndices,
            .vertexSize = sizeof(QuadVertex),
            .attributes = QuadVertex::getVertexInputAttributes(),
            .material = new QuadMaterial({ .baseColorTexture = &refFrameRT.colorBuffer ,}),
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

        // Setup masks for residual frame
        // We can use less vertices and indicies for the mask since it will be sparse
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

        // Setup depth mesh
        depthNode.setEntity(&depthMesh);
        depthNode.frustumCulled = false;
        depthNode.visible = false;
        depthNode.primativeType = GL_POINTS;

        // Setup hidden layers and wide fov RTs
        RenderTargetCreateParams rtParams = {
            .width = quadFrame.getSize().x,
            .height = quadFrame.getSize().y,
            .internalFormat = GL_RGBA16F,
            .format = GL_RGBA,
            .type = GL_HALF_FLOAT,
            .wrapS = GL_CLAMP_TO_EDGE,
            .wrapT = GL_CLAMP_TO_EDGE,
            .minFilter = GL_NEAREST,
            .magFilter = GL_NEAREST,
        };
        copyRTs.emplace_back(rtParams);
        for (int layer = 0; layer < numHidLayers; layer++) {
            copyRTs.emplace_back(rtParams);
            frameRTsHidLayer.emplace_back(rtParams);
        }

        for (int layer = 0; layer < numHidLayers; layer++) {
            meshParams.material = new QuadMaterial({ .baseColorTexture = &frameRTsHidLayer[layer].colorBuffer });
            // We can use less vertices and indicies for the hidden layers since they will be sparse
            meshParams.maxVertices = maxVertices / 4;
            meshParams.maxIndices = maxIndices / 4;
            meshesHidLayer.emplace_back(meshParams);

            nodesHidLayer.emplace_back(&meshesHidLayer[layer]);
            nodesHidLayer[layer].frustumCulled = false;

            const glm::vec4& color = colors[(layer + 1) % colors.size()];

            wireframesHidLayer.emplace_back(&meshesHidLayer[layer]);
            wireframesHidLayer[layer].frustumCulled = false;
            wireframesHidLayer[layer].wireframe = true;
            wireframesHidLayer[layer].overrideMaterial = new QuadMaterial({ .baseColor = color });

            depthMeshsHidLayer.emplace_back(quadFrame.getSize(), glm::vec4(0.0f, 1.0f, 0.0f, 1.0f));

            depthNodesHidLayer.emplace_back(&depthMeshsHidLayer[layer]);
            depthNodesHidLayer[layer].frustumCulled = false;
            depthNodesHidLayer[layer].primativeType = GL_POINTS;
        }

        // Setup scene to use as mask for wide fov camera
        for (int i = 0; i < 2; i++) {
            wideFovNodes.emplace_back(&refFrameMeshes[i]);
            wideFovNodes[i].frustumCulled = false;
            sceneWideFov.addChildNode(&wideFovNodes[i]);
        }
        for (int i = 0; i < numHidLayers - 1; i++) {
            sceneWideFov.addChildNode(&nodesHidLayer[i]);
        }
        sceneWideFov.addChildNode(&maskFrameNode);
    }
    ~QRSimulator() = default;

    void addMeshesToScene(Scene& localScene) {
        for (int i = 0; i < 2; i++) {
            localScene.addChildNode(&refFrameNodesLocal[i]);
            localScene.addChildNode(&refFrameWireframesLocal[i]);
        }
        localScene.addChildNode(&maskFrameNode);
        localScene.addChildNode(&maskFrameWireframeNodesLocal);
        localScene.addChildNode(&depthNode);

        for (int layer = 0; layer < maxLayers - 1; layer++) {
            localScene.addChildNode(&nodesHidLayer[layer]);
            localScene.addChildNode(&wireframesHidLayer[layer]);
            localScene.addChildNode(&depthNodesHidLayer[layer]);
        }
    }

    void generateFrame(
            const PerspectiveCamera& remoteCameraCenter, const PerspectiveCamera& remoteCameraWideFov, Scene& remoteScene,
            DeferredRenderer& remoteRenderer,
            DepthPeelingRenderer& remoteRendererDP,
            bool generateResFrame = false, bool showNormals = false, bool showDepth = false) {
        // Reset stats
        stats = { 0 };

        double startTime = timeutils::getTimeMicros();
        // Render remote scene with multiple layers
        remoteRendererDP.drawObjects(remoteScene, remoteCameraCenter);
        stats.totalRenderTime += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        for (int layer = 0; layer < maxLayers; layer++) {
            int hiddenIndex = layer - 1;
            auto& remoteCameraToUse = (layer == 0 && generateResFrame) ? remoteCameraPrev :
                                        ((layer != maxLayers - 1) ? remoteCameraCenter : remoteCameraWideFov);

            auto& frameToUse = (layer == 0) ? refFrameRT : frameRTsHidLayer[hiddenIndex];

            auto& meshToUse = (layer == 0) ? refFrameMeshes[currMeshIndex] : meshesHidLayer[hiddenIndex];
            auto& meshToUseDepth = (layer == 0) ? depthMesh : depthMeshsHidLayer[hiddenIndex];

            startTime = timeutils::getTimeMicros();
            if (layer == 0) {
                remoteRenderer.drawObjectsNoLighting(remoteScene, remoteCameraToUse);
                if (!showNormals) {
                    remoteRenderer.copyToFrameRT(frameToUse);
                    toneMapper.drawToRenderTarget(remoteRenderer, copyRTs[layer]);
                }
                else {
                    showNormalsEffect.drawToRenderTarget(remoteRenderer, frameToUse);
                }
            }
            else if (layer != maxLayers - 1) {
                // Copy to render target
                if (!showNormals) {
                    remoteRendererDP.peelingLayers[hiddenIndex+1].blitToFrameRT(frameToUse);
                    toneMapper.setUniforms(remoteRendererDP.peelingLayers[hiddenIndex+1]);
                    toneMapper.drawToRenderTarget(remoteRendererDP, copyRTs[layer], false);
                }
                else {
                    showNormalsEffect.drawToRenderTarget(remoteRendererDP, frameToUse);
                }
            }
            // Wide fov camera
            else {
                // Draw old center mesh at new remoteCamera layer, filling stencil buffer with 1
                remoteRenderer.pipeline.stencilState.enableRenderingIntoStencilBuffer(GL_KEEP, GL_KEEP, GL_REPLACE);
                remoteRenderer.pipeline.writeMaskState.disableColorWrites();
                wideFovNodes[currMeshIndex].visible = false;
                wideFovNodes[prevMeshIndex].visible = true;
                remoteRenderer.drawObjectsNoLighting(sceneWideFov, remoteCameraToUse);

                // Render remoteScene using stencil buffer as a mask
                // At values where stencil buffer is not 1, remoteScene should render
                remoteRenderer.pipeline.stencilState.enableRenderingUsingStencilBufferAsMask(GL_NOTEQUAL, 1);
                remoteRenderer.pipeline.writeMaskState.enableColorWrites();
                remoteRenderer.drawObjectsNoLighting(remoteScene, remoteCameraToUse, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                remoteRenderer.pipeline.stencilState.restoreStencilState();

                if (!showNormals) {
                    remoteRenderer.copyToFrameRT(frameToUse);
                    toneMapper.setUniforms(frameToUse);
                    toneMapper.drawToRenderTarget(remoteRenderer, copyRTs[layer], false);
                }
                else {
                    showNormalsEffect.drawToRenderTarget(remoteRenderer, frameToUse);
                }
            }
            stats.totalRenderTime += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

            /*
            ============================
            Generate Reference Frame
            ============================
            */
            uint numProxies = 0, numDepthOffsets = 0;
            auto& quadsGenerator = frameGenerator.quadsGenerator;
            auto oldParams = quadsGenerator.params;
            if (layer == maxLayers - 1) {
                quadsGenerator.params.maxIterForceMerge = 4;
                quadsGenerator.params.depthThreshold = 1e-3f;
                quadsGenerator.params.flattenThreshold = 0.5f;
                quadsGenerator.params.proxySimilarityThreshold = 5.0f;
            }
            else if (layer > 0) {
                quadsGenerator.params.maxIterForceMerge = 4;
                quadsGenerator.params.depthThreshold = 1e-3f;
            }
            quadsGenerator.params.expandEdges = false;
            uint numBytesIFrame = frameGenerator.generateRefFrame(
                frameToUse, remoteCameraToUse,
                meshToUse,
                numProxies, numDepthOffsets
            );
            if (!generateResFrame) {
                stats.compressedSizeBytes += numBytesIFrame;
            }
            quadsGenerator.params = oldParams;

            // Copy quads and depth offsets to local vectors
            quadFrame.copyToMemory(quads[layer], depthOffsets[layer]);

            stats.totalGenQuadMapTime += frameGenerator.stats.timeToGenerateQuadsMs;
            stats.totalSimplifyTime += frameGenerator.stats.timeToSimplifyQuadsMs;
            stats.totalGatherQuadsTime += frameGenerator.stats.timeToGatherQuadsMs;
            stats.totalCreateProxiesTime += frameGenerator.stats.timeToCreateProxiesMs;

            stats.totalAppendQuadsTime += frameGenerator.stats.timeToAppendQuadsMs;
            stats.totalFillQuadsIndiciesTime += frameGenerator.stats.timeToFillQuadIndicesMs;
            stats.totalCreateVertIndTime += frameGenerator.stats.timeToCreateVertIndMs;
            stats.totalCreateMeshTime += frameGenerator.stats.timeToCreateMeshMs;

            if (layer != 0 || !generateResFrame) {
                stats.totalCompressTime += frameGenerator.stats.timeToCompress;
            }

            /*
            ============================
            Generate Residual Frame
            ============================
            */
            if (layer == 0) {
                if (generateResFrame) {
                    quadsGenerator.params.expandEdges = true;
                    stats.compressedSizeBytes += frameGenerator.generateResFrame(
                        meshScenes[currMeshIndex], meshScenes[prevMeshIndex],
                        maskTempRT, maskFrameRT,
                        remoteCameraCenter, remoteCameraPrev,
                        refFrameMeshes[currMeshIndex], maskFrameMesh,
                        numProxies, numDepthOffsets
                    );

                    stats.totalRenderTime += frameGenerator.stats.timeToRenderMaskMs;

                    stats.totalGenQuadMapTime += frameGenerator.stats.timeToGenerateQuadsMs;
                    stats.totalSimplifyTime += frameGenerator.stats.timeToSimplifyQuadsMs;
                    stats.totalGatherQuadsTime += frameGenerator.stats.timeToGatherQuadsMs;
                    stats.totalCreateProxiesTime += frameGenerator.stats.timeToCreateProxiesMs;

                    stats.totalAppendQuadsTime += frameGenerator.stats.timeToAppendQuadsMs;
                    stats.totalFillQuadsIndiciesTime += frameGenerator.stats.timeToGatherQuadsMs;
                    stats.totalCreateVertIndTime += frameGenerator.stats.timeToCreateVertIndMs;
                    stats.totalCreateMeshTime += frameGenerator.stats.timeToCreateMeshMs;

                    stats.totalCompressTime += frameGenerator.stats.timeToCompress;
                }

                maskFrameNode.visible = generateResFrame;
                currMeshIndex = (currMeshIndex + 1) % 2;
                prevMeshIndex = (prevMeshIndex + 1) % 2;

                // Only update the previous camera pose if we are not generating a Residual Frame
                if (!generateResFrame) {
                    remoteCameraPrev.setViewMatrix(remoteCameraCenter.getViewMatrix());
                }
            }
            proxiesPerQuadSet[layer] = numProxies;
            depthOffsetsPerQuadSet[layer] = numDepthOffsets;
            stats.totalProxies += numProxies;
            stats.totalDepthOffsets += numDepthOffsets;

            // For debugging: Generate point cloud from depth map
            if (showDepth) {
                meshToUseDepth.update(remoteCameraToUse, frameToUse);
                stats.totalGenDepthTime += meshToUseDepth.stats.genDepthTime;
            }
        }
    }

    uint saveToFile(const Path& outputPath) {
        uint totalOutputSize = 0;
        for (int layer = 0; layer < maxLayers; layer++) {
            // Save quads
            double startTime = timeutils::getTimeMicros();
            Path filename = (outputPath / "quads").appendToName(std::to_string(layer)).withExtension(".bin.zstd");
            std::ofstream quadsFile = std::ofstream(filename, std::ios::binary);
            quadsFile.write(quads[layer].data(), quads[layer].size());
            quadsFile.close();
            spdlog::info("Saved {} quads ({:.3f}MB) in {:.3f}ms",
                         proxiesPerQuadSet[layer], static_cast<double>(quads[layer].size()) / BYTES_PER_MEGABYTE,
                            timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

            // Save depth offsets
            startTime = timeutils::getTimeMicros();
            Path offsetsFile = (outputPath / "depthOffsets").appendToName(std::to_string(layer)).withExtension(".bin.zstd");
            std::ofstream depthOffsetsFile = std::ofstream(offsetsFile, std::ios::binary);
            depthOffsetsFile.write(depthOffsets[layer].data(), depthOffsets[layer].size());
            depthOffsetsFile.close();
            spdlog::info("Saved {} depth offsets ({:.3f}MB) in {:.3f}ms",
                         depthOffsetsPerQuadSet[layer], static_cast<double>(depthOffsets[layer].size()) / BYTES_PER_MEGABYTE,
                            timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

            // Save color buffer
            Path colorFileName = outputPath / ("color" + std::to_string(layer));
            copyRTs[layer].saveColorAsJPG(colorFileName.withExtension(".jpg"));

            totalOutputSize += quads[layer].size() + depthOffsets[layer].size();
        }

        return totalOutputSize;
    }

private:
    QuadFrame& quadFrame;

    FrameGenerator& frameGenerator;

    std::vector<uint> proxiesPerQuadSet;
    std::vector<uint> depthOffsetsPerQuadSet;

    // Shaders
    ToneMapper toneMapper;
    ShowNormalsEffect showNormalsEffect;

    PerspectiveCamera remoteCameraPrev;

    // Scenes with resulting mesh
    std::vector<Scene> meshScenes;
    Scene sceneWideFov;

    QuadMaterial wireframeMaterial = QuadMaterial({. baseColor = colors[0] });
    QuadMaterial maskWireframeMaterial = QuadMaterial({. baseColor = colors[colors.size()-1] });
};

} // namespace quasar


#endif // QUASAR_SIMULATOR_H
