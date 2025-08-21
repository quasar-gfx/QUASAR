#ifndef QUASAR_SIMULATOR_H
#define QUASAR_SIMULATOR_H

#include <DepthMesh.h>
#include <Quads/FrameGenerator.h>
#include <PostProcessing/ToneMapper.h>
#include <PostProcessing/ShowNormalsEffect.h>

namespace quasar {

class QRSimulator {
public:
    uint maxLayers;

    uint maxVertices = MAX_NUM_PROXIES * NUM_SUB_QUADS * VERTICES_IN_A_QUAD;
    uint maxIndices = MAX_NUM_PROXIES * NUM_SUB_QUADS * INDICES_IN_A_QUAD;
    uint maxVerticesDepth;

    // Reference frame
    FrameRenderTarget refFrameRT;
    std::vector<QuadMesh> refFrameMeshes;
    std::vector<Node> refFrameNodes;
    std::vector<Node> refFrameWireframesLocal;
    int currMeshIndex = 0, prevMeshIndex = 1;
    std::vector<ReferenceFrame> referenceFrames;

    // Residual frame -- we only create the residuals to the visible layer
    FrameRenderTarget resFrameRT;
    QuadMesh resFrameMesh;
    Node resFrameNode;
    std::vector<ResidualFrame> residualFrames;

    // Local objects
    std::vector<Node> refFrameNodesLocal;
    Node resFrameWireframeNodesLocal;

    // Hidden layers
    std::vector<FrameRenderTarget> frameRTsHidLayer;
    std::vector<QuadMesh> meshesHidLayer;
    std::vector<Node> nodesHidLayer;
    std::vector<Node> wireframesHidLayer;

    // Depth point cloud for debugging
    std::vector<DepthMesh> depthMeshsHidLayer;
    std::vector<Node> depthNodesHidLayer;

    // Wide fov
    std::vector<Node> wideFovNodes;

    DepthMesh depthMesh;
    Node depthNode;

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
        QuadSet::Sizes totalSizes;
    } stats;

    QRSimulator(QuadSet& quadSet, uint maxLayers, const PerspectiveCamera& remoteCamera, FrameGenerator& frameGenerator)
        : quadSet(quadSet)
        , frameGenerator(frameGenerator)
        , maxLayers(maxLayers)
        , maxVerticesDepth(quadSet.getSize().x * quadSet.getSize().y)
        , refFrameRT({
            .width = quadSet.getSize().x,
            .height = quadSet.getSize().y,
            .internalFormat = GL_RGBA16F,
            .format = GL_RGBA,
            .type = GL_HALF_FLOAT,
            .wrapS = GL_CLAMP_TO_EDGE,
            .wrapT = GL_CLAMP_TO_EDGE,
            .minFilter = GL_NEAREST,
            .magFilter = GL_NEAREST,
        })
        , resFrameRT({
            .width = quadSet.getSize().x,
            .height = quadSet.getSize().y,
            .internalFormat = GL_RGBA16F,
            .format = GL_RGBA,
            .type = GL_HALF_FLOAT,
            .wrapS = GL_CLAMP_TO_EDGE,
            .wrapT = GL_CLAMP_TO_EDGE,
            .minFilter = GL_NEAREST,
            .magFilter = GL_NEAREST,
        })
        , depthMesh(quadSet.getSize(), glm::vec4(0.0f, 1.0f, 0.0f, 1.0f))
        // We can use less vertices and indicies for the mask since it will be sparse
        , resFrameMesh(quadSet, resFrameRT.colorTexture, MAX_NUM_PROXIES / 4)
        , wireframeMaterial({ .baseColor = colors[0] })
        , maskWireframeMaterial({ .baseColor = colors[colors.size()-1] })
    {
        remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());
        remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());

        meshScenes.resize(2);
        refFrameMeshes.reserve(2);
        refFrameNodes.reserve(2);
        wideFovNodes.reserve(2);
        refFrameNodesLocal.reserve(2);
        refFrameWireframesLocal.reserve(2);

        copyRTs.reserve(maxLayers);
        referenceFrames.resize(maxLayers);
        residualFrames.resize(maxLayers);

        uint numHidLayers = maxLayers - 1;

        frameRTsHidLayer.reserve(numHidLayers);
        meshesHidLayer.reserve(numHidLayers);
        depthMeshsHidLayer.reserve(numHidLayers);
        nodesHidLayer.reserve(numHidLayers);
        wireframesHidLayer.reserve(numHidLayers);
        depthNodesHidLayer.reserve(numHidLayers);

        // Setup visible layer for reference frame
        for (int i = 0; i < 2; i++) {
            refFrameMeshes.emplace_back(quadSet, refFrameRT.colorTexture);

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
        resFrameNode.setEntity(&resFrameMesh);
        resFrameNode.frustumCulled = false;

        resFrameWireframeNodesLocal.setEntity(&resFrameMesh);
        resFrameWireframeNodesLocal.frustumCulled = false;
        resFrameWireframeNodesLocal.wireframe = true;
        resFrameWireframeNodesLocal.visible = false;
        resFrameWireframeNodesLocal.overrideMaterial = &maskWireframeMaterial;

        // Setup depth mesh
        depthNode.setEntity(&depthMesh);
        depthNode.frustumCulled = false;
        depthNode.visible = false;
        depthNode.primativeType = GL_POINTS;

        // Setup hidden layers and wide fov RTs
        RenderTargetCreateParams rtParams = {
            .width = quadSet.getSize().x,
            .height = quadSet.getSize().y,
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
            // We can use less vertices and indicies for the hidden layers since they will be sparse
            meshesHidLayer.emplace_back(quadSet, frameRTsHidLayer[layer].colorTexture, MAX_NUM_PROXIES / 4);

            nodesHidLayer.emplace_back(&meshesHidLayer[layer]);
            nodesHidLayer[layer].frustumCulled = false;

            const glm::vec4& color = colors[(layer + 1) % colors.size()];

            wireframesHidLayer.emplace_back(&meshesHidLayer[layer]);
            wireframesHidLayer[layer].frustumCulled = false;
            wireframesHidLayer[layer].wireframe = true;
            wireframesHidLayer[layer].overrideMaterial = new QuadMaterial({ .baseColor = color });

            depthMeshsHidLayer.emplace_back(quadSet.getSize(), glm::vec4(0.0f, 1.0f, 0.0f, 1.0f));
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
        sceneWideFov.addChildNode(&resFrameNode);
    }
    ~QRSimulator() = default;

    uint getNumTriangles() const {
        auto refMeshSizes = refFrameMeshes[currMeshIndex].getBufferSizes();
        uint numTriangles = refMeshSizes.numIndices / 3; // Each triangle has 3 indices
        for (const auto& mesh : meshesHidLayer) {
            auto size = mesh.getBufferSizes();
            numTriangles += size.numIndices / 3; // Each triangle has 3 indices
        }
        return numTriangles;
    }

    void addMeshesToScene(Scene& localScene) {
        for (int i = 0; i < 2; i++) {
            localScene.addChildNode(&refFrameNodesLocal[i]);
            localScene.addChildNode(&refFrameWireframesLocal[i]);
        }
        localScene.addChildNode(&resFrameNode);
        localScene.addChildNode(&resFrameWireframeNodesLocal);
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
        bool generateResFrame = false, bool showNormals = false, bool showDepth = false)
    {
        // Reset stats
        stats = { 0 };

        // Render remote scene with multiple layers
        double startTime = timeutils::getTimeMicros();
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
                    toneMapper.setUniforms(frameToUse);
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
            frameGenerator.generateRefFrame(
                frameToUse, remoteCameraToUse,
                meshToUse,
                referenceFrames[layer]
            );

            quadsGenerator.params = oldParams;

            stats.totalGenQuadMapTime += frameGenerator.stats.timeToGenerateQuadsMs;
            stats.totalSimplifyTime += frameGenerator.stats.timeToSimplifyQuadsMs;
            stats.totalGatherQuadsTime += frameGenerator.stats.timeToGatherQuadsMs;
            stats.totalCreateProxiesTime += frameGenerator.stats.timeToCreateQuadsMs;

            stats.totalAppendQuadsTime += frameGenerator.stats.timeToAppendQuadsMs;
            stats.totalFillQuadsIndiciesTime += frameGenerator.stats.timeToFillQuadIndicesMs;
            stats.totalCreateVertIndTime += frameGenerator.stats.timeToCreateVertIndMs;
            stats.totalCreateMeshTime += frameGenerator.stats.timeToCreateMeshMs;

            if (layer != 0 || !generateResFrame) {
                stats.totalCompressTime += frameGenerator.stats.timeToCompressMs;
            }

            /*
            ============================
            Generate Residual Frame
            ============================
            */
            if (layer == 0) {
                if (generateResFrame) {
                    quadsGenerator.params.expandEdges = true;
                    frameGenerator.generateResFrame(
                        meshScenes[currMeshIndex], meshScenes[prevMeshIndex],
                        resFrameRT,
                        remoteCameraCenter, remoteCameraPrev,
                        refFrameMeshes[currMeshIndex], resFrameMesh,
                        residualFrames[layer]
                    );

                    stats.totalRenderTime += frameGenerator.stats.timeToRenderMaskMs;

                    stats.totalGenQuadMapTime += frameGenerator.stats.timeToGenerateQuadsMs;
                    stats.totalSimplifyTime += frameGenerator.stats.timeToSimplifyQuadsMs;
                    stats.totalGatherQuadsTime += frameGenerator.stats.timeToGatherQuadsMs;
                    stats.totalCreateProxiesTime += frameGenerator.stats.timeToCreateQuadsMs;

                    stats.totalAppendQuadsTime += frameGenerator.stats.timeToAppendQuadsMs;
                    stats.totalFillQuadsIndiciesTime += frameGenerator.stats.timeToGatherQuadsMs;
                    stats.totalCreateVertIndTime += frameGenerator.stats.timeToCreateVertIndMs;
                    stats.totalCreateMeshTime += frameGenerator.stats.timeToCreateMeshMs;

                    stats.totalCompressTime += frameGenerator.stats.timeToCompressMs;
                }

                resFrameNode.visible = generateResFrame;
                currMeshIndex = (currMeshIndex + 1) % meshScenes.size();
                prevMeshIndex = (prevMeshIndex + 1) % meshScenes.size();

                // Only update the previous camera pose if we are not generating a Residual Frame
                if (!generateResFrame) {
                    remoteCameraPrev.setViewMatrix(remoteCameraCenter.getViewMatrix());
                }
            }

            // For debugging: Generate point cloud from depth map
            if (showDepth) {
                meshToUseDepth.update(remoteCameraToUse, frameToUse);
                stats.totalGenDepthTime += meshToUseDepth.stats.genDepthTime;
            }

            if (!generateResFrame) {
                stats.totalSizes.numQuads += referenceFrames[layer].getTotalNumQuads();
                stats.totalSizes.numDepthOffsets += referenceFrames[layer].getTotalNumDepthOffsets();
                stats.totalSizes.quadsSize += referenceFrames[layer].getTotalQuadsSize();
                stats.totalSizes.depthOffsetsSize += referenceFrames[layer].getTotalDepthOffsetsSize();
            }
            else {
                stats.totalSizes.numQuads += residualFrames[layer].getTotalNumQuads();
                stats.totalSizes.numDepthOffsets += residualFrames[layer].getTotalNumDepthOffsets();
                stats.totalSizes.quadsSize += residualFrames[layer].getTotalQuadsSize();
                stats.totalSizes.depthOffsetsSize += residualFrames[layer].getTotalDepthOffsetsSize();
            }
        }
    }

    uint saveToFile(const Path& outputPath) {
        uint totalOutputSize = 0;
        for (int layer = 0; layer < maxLayers; layer++) {
            // Save color
            Path colorFileName = outputPath / ("color" + std::to_string(layer));
            copyRTs[layer].saveColorAsJPG(colorFileName.withExtension(".jpg"));

            // Save proxies
            totalOutputSize += referenceFrames[layer].saveToFiles(outputPath);
        }
        return totalOutputSize;
    }

private:
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

    QuadSet& quadSet;
    FrameGenerator& frameGenerator;

    PerspectiveCamera remoteCameraPrev;

    // Scenes with resulting meshes
    std::vector<Scene> meshScenes;
    Scene sceneWideFov;

    // Holds a copies of the current frame
    std::vector<FrameRenderTarget> copyRTs;

    // Shaders
    ToneMapper toneMapper;
    ShowNormalsEffect showNormalsEffect;

    QuadMaterial wireframeMaterial;
    QuadMaterial maskWireframeMaterial;
};

} // namespace quasar

#endif // QUASAR_SIMULATOR_H
