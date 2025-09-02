#include <Streamers/QUASARStreamer.h>

using namespace quasar;

QUASARStreamer::QUASARStreamer(
        QuadSet& quadSet,
        uint maxLayers,
        DeferredRenderer& remoteRenderer,
        Scene& remoteScene,
        const PerspectiveCamera& remoteCamera,
        float wideFOV,
        const std::string& videoURL,
        const std::string& proxiesURL)
    : videoURL(videoURL)
    , proxiesURL(proxiesURL)
    , quadSet(quadSet)
    , maxLayers(maxLayers)
    , remoteRenderer(remoteRenderer)
    , remoteScene(remoteScene)
    , frameGenerator(quadSet)
    , referenceFrameRT({
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
    , referenceFrameRT_noTone({
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
    , residualFrameMaskRT({
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
    , residualFrameRT({
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
    , residualFrameRT_noTone({
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
    , atlasVideoStreamerRT({
        .width = 2 * quadSet.getSize().x,
        .height = 3 * quadSet.getSize().y,
        .internalFormat = GL_SRGB8,
        .format = GL_RGB,
        .type = GL_UNSIGNED_BYTE,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    }, videoURL)
    , depthMesh(quadSet.getSize(), glm::vec4(0.0f, 1.0f, 0.0f, 1.0f))
    // We can use less vertices and indicies for the mask since it will be sparse
    , residualFrameMesh(quadSet, residualFrameRT_noTone.colorTexture, MAX_QUADS_PER_MESH / 4)
    , wireframeMaterial({ .baseColor = colors[0] })
    , maskWireframeMaterial({ .baseColor = colors[colors.size()-1] })
    , DataStreamerTCP(proxiesURL)
{
    meshScenes.resize(2);
    referenceFrameMeshes.reserve(meshScenes.size());
    referenceFrameNodes.reserve(meshScenes.size());
    wideFovNodes.reserve(meshScenes.size());
    referenceFrameNodesLocal.reserve(meshScenes.size());
    referenceFrameWireframesLocal.reserve(meshScenes.size());

    referenceFrames.resize(maxLayers);
    residualFrames.resize(maxLayers);

    uint numHidLayers = maxLayers - 1;
    frameRTsHidLayer.reserve(numHidLayers);
    frameRTsHidLayer_noTone.reserve(numHidLayers);
    meshesHidLayer.reserve(numHidLayers);
    depthMeshsHidLayer.reserve(numHidLayers);
    nodesHidLayer.reserve(numHidLayers);
    wireframesHidLayer.reserve(numHidLayers);
    depthNodesHidLayer.reserve(numHidLayers);

    remoteCameraWideFOV.setProjectionMatrix(remoteCamera.getProjectionMatrix());
    remoteCameraWideFOV.setFovyDegrees(wideFOV);
    remoteCameraWideFOV.setViewMatrix(remoteCamera.getViewMatrix());

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
    for (int layer = 0; layer < numHidLayers; layer++) {
        frameRTsHidLayer.emplace_back(rtParams);
        frameRTsHidLayer_noTone.emplace_back(rtParams);
    }

    // Setup visible layer for reference frame
    for (int i = 0; i < meshScenes.size(); i++) {
        referenceFrameMeshes.emplace_back(quadSet, referenceFrameRT_noTone.colorTexture);

        referenceFrameNodes.emplace_back(&referenceFrameMeshes[i]);
        referenceFrameNodes[i].frustumCulled = false;
        meshScenes[i].addChildNode(&referenceFrameNodes[i]);

        referenceFrameNodesLocal.emplace_back(&referenceFrameMeshes[i]);
        referenceFrameNodesLocal[i].frustumCulled = false;

        referenceFrameWireframesLocal.emplace_back(&referenceFrameMeshes[i]);
        referenceFrameWireframesLocal[i].frustumCulled = false;
        referenceFrameWireframesLocal[i].wireframe = true;
        referenceFrameWireframesLocal[i].visible = false;
        referenceFrameWireframesLocal[i].overrideMaterial = &wireframeMaterial;
    }

    // Setup masks for residual frame
    residualFrameNode.setEntity(&residualFrameMesh);
    residualFrameNode.frustumCulled = false;

    residualFrameWireframesLocal.setEntity(&residualFrameMesh);
    residualFrameWireframesLocal.frustumCulled = false;
    residualFrameWireframesLocal.wireframe = true;
    residualFrameWireframesLocal.visible = false;
    residualFrameWireframesLocal.overrideMaterial = &maskWireframeMaterial;

    // Setup depth mesh
    depthNode.setEntity(&depthMesh);
    depthNode.frustumCulled = false;
    depthNode.visible = false;
    depthNode.primitiveType = GL_POINTS;

    for (int layer = 0; layer < numHidLayers; layer++) {
        // We can use less vertices and indicies for the hidden layers since they will be sparser
        meshesHidLayer.emplace_back(quadSet, frameRTsHidLayer_noTone[layer].colorTexture, MAX_QUADS_PER_MESH / 4);

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
        depthNodesHidLayer[layer].primitiveType = GL_POINTS;
    }

    // Setup scene to use as mask for wide fov camera
    for (int i = 0; i < meshScenes.size(); i++) {
        wideFovNodes.emplace_back(&referenceFrameMeshes[i]);
        wideFovNodes[i].frustumCulled = false;
        sceneWideFov.addChildNode(&wideFovNodes[i]);
    }
    for (int i = 0; i < numHidLayers - 1; i++) {
        sceneWideFov.addChildNode(&nodesHidLayer[i]);
    }
    sceneWideFov.addChildNode(&residualFrameNode);
}

uint QUASARStreamer::getNumTriangles() const {
    auto refMeshSizes = referenceFrameMeshes[currMeshIndex].getBufferSizes();
    uint numTriangles = refMeshSizes.numIndices / 3; // Each triangle has 3 indices
    for (const auto& mesh : meshesHidLayer) {
        auto size = mesh.getBufferSizes();
        numTriangles += size.numIndices / 3; // Each triangle has 3 indices
    }
    return numTriangles;
}

void QUASARStreamer::addMeshesToScene(Scene& localScene) {
    for (int i = 0; i < meshScenes.size(); i++) {
        localScene.addChildNode(&referenceFrameNodesLocal[i]);
        localScene.addChildNode(&referenceFrameWireframesLocal[i]);
    }
    localScene.addChildNode(&residualFrameNode);
    localScene.addChildNode(&residualFrameWireframesLocal);
    localScene.addChildNode(&depthNode);

    for (int layer = 0; layer < maxLayers - 1; layer++) {
        localScene.addChildNode(&nodesHidLayer[layer]);
        localScene.addChildNode(&wireframesHidLayer[layer]);
        localScene.addChildNode(&depthNodesHidLayer[layer]);
    }
}

void QUASARStreamer::updateViewSphere(const PerspectiveCamera& remoteCamera, float viewSphereDiameter) {
    this->viewSphereDiameter = viewSphereDiameter;

    // Update wide FOV camera
    remoteCameraWideFOV.setViewMatrix(remoteCamera.getViewMatrix());
}

void QUASARStreamer::generateFrame(
    DeferredRenderer& remoteRenderer, DepthPeelingRenderer& remoteRendererDP,
    Scene& remoteScene, const PerspectiveCamera& remoteCamera,
    bool createResidualFrame, bool showNormals, bool showDepth)
{
    // Reset stats
    stats = { 0 };

    auto quadsGenerator = frameGenerator.getQuadsGenerator();

    if (!createResidualFrame) {
        std::swap(currMeshIndex, prevMeshIndex);
    }

    /*
    ============================
    Render scene normally to create Reference Frame textures
    ============================
    */
    double startTime = timeutils::getTimeMicros();
    remoteRendererDP.drawObjects(remoteScene, remoteCamera);
    stats.totalRenderTime += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    for (int layer = 0; layer < maxLayers; layer++) {
        int hiddenLayerIndex = layer - 1;
        auto& remoteCameraToUse = (layer == 0 && createResidualFrame) ? remoteCameraPrev :
                                  ((layer != maxLayers - 1) ? remoteCamera : remoteCameraWideFOV);

        auto& frameToUse = (layer == 0) ? referenceFrameRT : frameRTsHidLayer[hiddenLayerIndex];
        auto& frameToUse_noTone = (layer == 0) ? referenceFrameRT_noTone : frameRTsHidLayer_noTone[hiddenLayerIndex];

        auto& meshToUse = (layer == 0) ? referenceFrameMeshes[currMeshIndex] : meshesHidLayer[hiddenLayerIndex];
        auto& meshToUseDepth = (layer == 0) ? depthMesh : depthMeshsHidLayer[hiddenLayerIndex];

        startTime = timeutils::getTimeMicros();
        if (layer == 0) {
            remoteRenderer.drawObjectsNoLighting(remoteScene, remoteCameraToUse);
            remoteRenderer.copyToFrameRT(frameToUse);
        }
        else if (layer < maxLayers - 1) {
            // Hidden layers need to use the noTone render targets to generate quads for some reason...
            remoteRendererDP.peelingLayers[hiddenLayerIndex+1].blit(frameToUse_noTone);
        }
        // Wide fov camera
        else {
            // Draw old center mesh at new remoteCamera layer, filling stencil buffer with 1
            remoteRenderer.pipeline.stencilState.enableRenderingIntoStencilBuffer(GL_KEEP, GL_KEEP, GL_REPLACE);
            remoteRenderer.pipeline.writeMaskState.disableColorWrites();
            wideFovNodes[currMeshIndex].visible = true;
            wideFovNodes[prevMeshIndex].visible = false;
            remoteRenderer.drawObjectsNoLighting(sceneWideFov, remoteCameraToUse);

            // Render remoteScene using stencil buffer as a mask
            // At values where stencil buffer is not 1, remoteScene should render
            remoteRenderer.pipeline.stencilState.enableRenderingUsingStencilBufferAsMask(GL_NOTEQUAL, 1);
            remoteRenderer.pipeline.writeMaskState.enableColorWrites();
            remoteRenderer.drawObjectsNoLighting(remoteScene, remoteCameraToUse, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            remoteRenderer.pipeline.stencilState.restoreStencilState();
            remoteRenderer.copyToFrameRT(frameToUse);
        }
        stats.totalRenderTime += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        /*
        ============================
        Generate Reference Frame
        ============================
        */
        auto oldParams = quadsGenerator->params;
        if (layer == maxLayers - 1) {
            quadsGenerator->params.depthThreshold = 1e-3f;
            quadsGenerator->params.flattenThreshold = 0.5f;
            quadsGenerator->params.proxySimilarityThreshold = 5.0f;
            quadsGenerator->params.maxIterForceMerge = 4;
        }
        else if (layer > 0) {
            quadsGenerator->params.depthThreshold = 1e-3f;
            quadsGenerator->params.maxIterForceMerge = 4;
        }
        quadsGenerator->params.expandEdges = false;
        ReferenceFrame currReferenceFrame;
        frameGenerator.createReferenceFrame(
            (layer != 0 && layer != maxLayers - 1) ? frameToUse_noTone : frameToUse,
            remoteCameraToUse,
            meshToUse,
            ((layer == 0) && createResidualFrame) ? currReferenceFrame : referenceFrames[layer]
        );
        if (!showNormals) {
            if (layer == 0) {
                remoteRenderer.copyToFrameRT(frameToUse_noTone);
                toneMapper.drawToRenderTarget(remoteRenderer, frameToUse);
            }
            else if (layer < maxLayers - 1) {
                toneMapper.setUniforms(frameToUse_noTone);
                toneMapper.drawToRenderTarget(remoteRenderer, frameToUse, false);
            }
            else {
                remoteRenderer.copyToFrameRT(frameToUse_noTone);
                toneMapper.drawToRenderTarget(remoteRenderer, frameToUse);
            }
        }
        else {
            showNormalsEffect.drawToRenderTarget(remoteRenderer, frameToUse_noTone);
        }

        quadsGenerator->params = oldParams;

        stats.totalGenQuadMapTime += frameGenerator.stats.timeToGenerateQuadsMs;
        stats.totalSimplifyTime += frameGenerator.stats.timeToSimplifyQuadsMs;
        stats.totalGatherQuadsTime += frameGenerator.stats.timeToGatherQuadsMs;
        stats.totalCreateProxiesTime += frameGenerator.stats.timeToCreateQuadsMs;

        stats.totalAppendQuadsTime += frameGenerator.stats.timeToAppendQuadsMs;
        stats.totalFillQuadsIndiciesTime += frameGenerator.stats.timeToFillQuadIndicesMs;
        stats.totalCreateVertIndTime += frameGenerator.stats.timeToCreateVertIndMs;
        stats.totaltimeToCreateMeshMs += frameGenerator.stats.timeToCreateMeshMs;

        if (layer != 0 || !createResidualFrame) {
            stats.totalCompressTime += frameGenerator.stats.timeToCompressMs;
        }

        /*
        ============================
        Generate Residual Frame
        ============================
        */
        if (layer == 0) {
            if (createResidualFrame) {
                /*
                ============================
                Generate masked Residual Frame textures
                ============================
                */
                frameGenerator.updateResidualRenderTargets(
                    residualFrameMaskRT, residualFrameRT,
                    remoteRenderer, remoteScene,
                    meshScenes[currMeshIndex], meshScenes[prevMeshIndex],
                    remoteCamera, remoteCameraPrev
                );

                /*
                ============================
                Generate Residual Frame
                ============================
                */
                quadsGenerator->params.expandEdges = true;
                frameGenerator.createResidualFrame(
                    residualFrameMaskRT, residualFrameRT,
                    remoteCamera, remoteCameraPrev,
                    referenceFrameMeshes[prevMeshIndex], residualFrameMesh,
                    residualFrames[layer]
                );
                if (!showNormals) {
                    residualFrameRT.blit(residualFrameRT_noTone);
                    toneMapper.setUniforms(residualFrameRT_noTone);
                    toneMapper.drawToRenderTarget(remoteRenderer, residualFrameRT, false);
                }
                else {
                    showNormalsEffect.drawToRenderTarget(remoteRenderer, residualFrameRT_noTone);
                }

                stats.totalRenderTime += frameGenerator.stats.timeToUpdateRTsMs;

                stats.totalGenQuadMapTime += frameGenerator.stats.timeToGenerateQuadsMs;
                stats.totalSimplifyTime += frameGenerator.stats.timeToSimplifyQuadsMs;
                stats.totalGatherQuadsTime += frameGenerator.stats.timeToGatherQuadsMs;
                stats.totalCreateProxiesTime += frameGenerator.stats.timeToCreateQuadsMs;

                stats.totalAppendQuadsTime += frameGenerator.stats.timeToAppendQuadsMs;
                stats.totalFillQuadsIndiciesTime += frameGenerator.stats.timeToGatherQuadsMs;
                stats.totalCreateVertIndTime += frameGenerator.stats.timeToCreateVertIndMs;
                stats.totaltimeToCreateMeshMs += frameGenerator.stats.timeToCreateMeshMs;

                stats.totalCompressTime += frameGenerator.stats.timeToCompressMs;
            }

            residualFrameNode.visible = createResidualFrame;

            // Only update the previous camera pose if we are not generating a Residual Frame
            if (!createResidualFrame) {
                remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());
            }
        }

        // For debugging: Generate point cloud from depth map
        if (showDepth) {
            meshToUseDepth.update(remoteCameraToUse, frameToUse);
            stats.totalGenDepthTime += meshToUseDepth.stats.genDepthTime;
        }

        if (!createResidualFrame) {
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

    // Update texture atlas (tile frames side by side)
    uint row = 0, col = 0;
    uint dstWidth = referenceFrameRT.width, dstHeight = referenceFrameRT.height;
    for (int layer = 0; layer < maxLayers; layer++) {
        if (layer == 0) {
            referenceFrameRT.blit(atlasVideoStreamerRT,
                0, 0, referenceFrameRT.width, referenceFrameRT.height,
                col, row, dstWidth, dstHeight
            );
        }
        else {
            int hiddenLayerIndex = layer - 1;
            frameRTsHidLayer[hiddenLayerIndex].blit(atlasVideoStreamerRT,
                0, 0, frameRTsHidLayer[hiddenLayerIndex].width, frameRTsHidLayer[hiddenLayerIndex].height,
                col, row, dstWidth, dstHeight
            );
        }
        col += referenceFrameRT.width;
        dstWidth += referenceFrameRT.width;
        if (col >= atlasVideoStreamerRT.width) {
            col = 0;
            dstWidth = referenceFrameRT.width;

            row += referenceFrameRT.height;
            dstHeight += referenceFrameRT.height;
            if (row >= atlasVideoStreamerRT.height) {
                row = 0;
                dstHeight = referenceFrameRT.height;
            }
        }
    }
    if (createResidualFrame) {
        residualFrameRT.blit(atlasVideoStreamerRT,
            0, 0, residualFrameRT.width, residualFrameRT.height,
            col, row, dstWidth, dstHeight
        );
    }
}

size_t QUASARStreamer::writeToFiles(const PerspectiveCamera& remoteCamera, const Path& outputPath) {
    // Save camera data
    Pose cameraPose;
    Path cameraFileName = (outputPath / "camera").withExtension(".bin");
    cameraPose.setProjectionMatrix(remoteCamera.getProjectionMatrix());
    cameraPose.setViewMatrix(remoteCamera.getViewMatrix());
    cameraPose.writeToFile(cameraFileName);

    // Save metadata (viewSphereDiameter and wide FOV)
    std::vector<float> metadata = {
        remoteCameraWideFOV.getFovyDegrees(),
        viewSphereDiameter,
    };
    FileIO::writeToBinaryFile(outputPath / "metadata.bin", metadata.data(), metadata.size() * sizeof(float));

    // Save color
    Path colorFileName = (outputPath / "color").withExtension(".jpg");
    atlasVideoStreamerRT.writeColorAsJPG(colorFileName);

    size_t totalOutputSize = 0;
    for (int layer = 0; layer < maxLayers; layer++) {
        // Save proxies
        totalOutputSize += referenceFrames[layer].writeToFiles(outputPath, layer);
    }
    return totalOutputSize;
}
