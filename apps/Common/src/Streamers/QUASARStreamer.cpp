#include <Streamers/QUASARStreamer.h>

using namespace quasar;

QUASARStreamer::QUASARStreamer(
        QuadSet& quadSet,
        DepthPeelingRenderer& remoteRendererDP,
        DeferredRenderer& remoteRenderer,
        Scene& remoteScene,
        PerspectiveCamera& remoteCamera,
        const QUASARStreamerCreateParams& params)
    : quadSet(quadSet)
    , videoURL(params.videoURL)
    , proxiesURL(params.proxiesURL)
    , maxLayers(params.maxLayers)
    , remoteRenderer(remoteRenderer)
    , remoteRendererDP(remoteRendererDP)
    , remoteScene(remoteScene)
    , remoteCamera(remoteCamera)
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
        .internalFormat = GL_SRGB8_ALPHA8,
        .format = GL_RGBA,
        .type = GL_UNSIGNED_BYTE,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    }, params.videoURL, params.targetFramerate, params.targetBitRate)
    , depthMesh(quadSet.getSize(), glm::vec4(0.0f, 1.0f, 0.0f, 1.0f))
    , residualFrameMesh(quadSet, residualFrameRT_noTone.colorTexture)
    , wireframeMaterial({ .baseColor = colors[0] })
    , maskWireframeMaterial({ .baseColor = colors[colors.size()-1] })
    , DataStreamerTCP(params.proxiesURL)
{
    meshScenes.resize(2);
    referenceFrameMeshes.reserve(meshScenes.size());
    referenceFrameNodes.reserve(meshScenes.size());
    wideFovNodes.reserve(meshScenes.size());
    referenceFrameNodesLocal.reserve(meshScenes.size());
    referenceFrameWireframesLocal.reserve(meshScenes.size());

    referenceFrames.resize(maxLayers);
    geometryMetadatas.resize(maxLayers);

    uint numHidLayers = maxLayers - 1;
    frameRTsHidLayer.reserve(numHidLayers);
    frameRTsHidLayer_noTone.reserve(numHidLayers);
    meshesHidLayer.reserve(numHidLayers);
    depthMeshsHidLayer.reserve(numHidLayers);
    nodesHidLayer.reserve(numHidLayers);
    wireframesHidLayer.reserve(numHidLayers);
    depthNodesHidLayer.reserve(numHidLayers);

    remoteCameraPrev.setProjectionMatrix(remoteCamera.getProjectionMatrix());
    remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());

    remoteCameraWideFOV.setProjectionMatrix(remoteCamera.getProjectionMatrix());
    remoteCameraWideFOV.setFovyDegrees(params.wideFOV);
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
        meshesHidLayer.emplace_back(quadSet, frameRTsHidLayer_noTone[layer].colorTexture);

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

    setViewSphereDiameter(params.viewSphereDiameter);

    if (!videoURL.empty() && !proxiesURL.empty()) {
        spdlog::info("Created QUASARStreamer that sends to URL: tcp://{}", proxiesURL);
    }
}

QUASARStreamer::~QUASARStreamer() {
    atlasVideoStreamerRT.stop();
}

uint QUASARStreamer::getNumTriangles() const {
    int currMeshIndex  = meshIndex % 2;
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

void QUASARStreamer::setViewSphereDiameter(float viewSphereDiameter) {
    this->viewSphereDiameter = viewSphereDiameter;
    remoteRendererDP.setViewSphereDiameter(viewSphereDiameter);
}

RenderStats QUASARStreamer::generateFrame(bool createResidualFrame, bool showNormals, bool showDepth) {
    // Reset stats
    stats = { 0 };

    int currMeshIndex  = meshIndex % 2;
    int prevMeshIndex  = (meshIndex + 1) % 2;

    // Update wide FOV camera
    remoteCameraWideFOV.setViewMatrix(remoteCamera.getViewMatrix());

    auto quadsGenerator = frameGenerator.getQuadsGenerator();

    /*
    ============================
    Render scene normally to create Reference Frame textures
    ============================
    */

    double startTime = timeutils::getTimeMicros();
    RenderStats renderStats = remoteRendererDP.drawObjects(remoteScene, remoteCamera);
    stats.totalRenderTimeMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

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
            renderStats += remoteRenderer.drawObjectsNoLighting(remoteScene, remoteCameraToUse);
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
            renderStats += remoteRenderer.drawObjectsNoLighting(sceneWideFov, remoteCameraToUse);

            // Render remoteScene using stencil buffer as a mask
            // At values where stencil buffer is not 1, remoteScene should render
            remoteRenderer.pipeline.stencilState.enableRenderingUsingStencilBufferAsMask(GL_NOTEQUAL, 1);
            remoteRenderer.pipeline.writeMaskState.enableColorWrites();
            renderStats += remoteRenderer.drawObjectsNoLighting(remoteScene, remoteCameraToUse, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            remoteRenderer.pipeline.stencilState.restoreStencilState();
            remoteRenderer.copyToFrameRT(frameToUse);
        }
        stats.totalRenderTimeMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

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
        ReferenceFrame dummyFrame;
        frameGenerator.createReferenceFrame(
            (layer != 0 && layer != maxLayers - 1) ? frameToUse_noTone : frameToUse,
            remoteCameraToUse,
            meshToUse,
            (layer == 0 && createResidualFrame) ? dummyFrame : referenceFrames[layer] // Don't save output of this reference frame
        );
        if (!showNormals) {
            if (layer == 0) {
                remoteRenderer.copyToFrameRT(frameToUse_noTone);
                tonemapper.drawToRenderTarget(remoteRenderer, frameToUse);
            }
            else if (layer < maxLayers - 1) {
                tonemapper.setUniforms(frameToUse_noTone);
                tonemapper.drawToRenderTarget(remoteRenderer, frameToUse, false);
            }
            else {
                remoteRenderer.copyToFrameRT(frameToUse_noTone);
                tonemapper.drawToRenderTarget(remoteRenderer, frameToUse);
            }
        }
        else {
            showNormalsEffect.drawToRenderTarget(remoteRenderer, frameToUse_noTone);
        }

        quadsGenerator->params = oldParams;

        stats.totalGenQuadMapTimeMs += frameGenerator.stats.generateQuadsTimeMs;
        stats.totalSimplifyTimeMs += frameGenerator.stats.simplifyQuadsTimeMs;
        stats.totalGatherQuadsTime += frameGenerator.stats.gatherQuadsTimeMs;
        stats.totalCreateProxiesTimeMs += frameGenerator.stats.createQuadsTimeMs;

        stats.totalAppendQuadsTimeMs += frameGenerator.stats.appendQuadsTimeMs;
        stats.totalCreateVertIndTimeMs += frameGenerator.stats.createVertIndTimeMs;
        stats.totalCreateMeshTimeMs += frameGenerator.stats.createMeshTimeMs;

        if (!createResidualFrame || layer != 0) {
            stats.totalCompressTimeMs += frameGenerator.stats.compressTimeMs;
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
                    residualFrame
                );
                if (!showNormals) {
                    residualFrameRT.blit(residualFrameRT_noTone);
                    tonemapper.setUniforms(residualFrameRT_noTone);
                    tonemapper.drawToRenderTarget(remoteRenderer, residualFrameRT, false);
                }
                else {
                    showNormalsEffect.drawToRenderTarget(remoteRenderer, residualFrameRT_noTone);
                }

                stats.totalRenderTimeMs += frameGenerator.stats.updateRTsTimeMs;

                stats.totalGenQuadMapTimeMs += frameGenerator.stats.generateQuadsTimeMs;
                stats.totalSimplifyTimeMs += frameGenerator.stats.simplifyQuadsTimeMs;
                stats.totalGatherQuadsTime += frameGenerator.stats.gatherQuadsTimeMs;
                stats.totalCreateProxiesTimeMs += frameGenerator.stats.createQuadsTimeMs;

                stats.totalAppendQuadsTimeMs += frameGenerator.stats.appendQuadsTimeMs;
                stats.totalCreateVertIndTimeMs += frameGenerator.stats.createVertIndTimeMs;
                stats.totalCreateMeshTimeMs += frameGenerator.stats.createMeshTimeMs;

                stats.totalCompressTimeMs += frameGenerator.stats.compressTimeMs;
            }
            else {
                // Only update the previous camera pose if we are not generating a Residual Frame
                remoteCameraPrev.setProjectionMatrix(remoteCamera.getProjectionMatrix());
                remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());
                lastMeshIndex = meshIndex;
                meshIndex++;
            }
        }

        residualFrameNode.visible = createResidualFrame;

        // For debugging: Generate point cloud from depth map
        if (showDepth) {
            meshToUseDepth.update(remoteCameraToUse, frameToUse);
            stats.totalGenDepthTimeMs += meshToUseDepth.stats.genDepthTime;
        }

        if (!(createResidualFrame && layer == 0)) {
            stats.totalSizes.numQuads += referenceFrames[layer].getTotalNumQuads();
            stats.totalSizes.numDepthOffsets += referenceFrames[layer].getTotalNumDepthOffsets();
            stats.totalSizes.quadsSize += referenceFrames[layer].getTotalQuadsSize();
            stats.totalSizes.depthOffsetsSize += referenceFrames[layer].getTotalDepthOffsetsSize();
            spdlog::debug("Reference frame generated with {} quads ({:.3f} MB), {} depth offsets ({:.3f} MB)",
                          referenceFrames[layer].getTotalNumQuads(), referenceFrames[layer].getTotalQuadsSize() / BYTES_PER_MEGABYTE,
                          referenceFrames[layer].getTotalNumDepthOffsets(), referenceFrames[layer].getTotalDepthOffsetsSize() / BYTES_PER_MEGABYTE);
        }
        else {
            stats.totalSizes.numQuads += residualFrame.getTotalNumQuads();
            stats.totalSizes.numDepthOffsets += residualFrame.getTotalNumDepthOffsets();
            stats.totalSizes.quadsSize += residualFrame.getTotalQuadsSize();
            stats.totalSizes.depthOffsetsSize += residualFrame.getTotalDepthOffsetsSize();
            spdlog::debug("Residual frame generated with {} quads ({:.3f} MB), {} depth offsets ({:.3f} MB)",
                          residualFrame.getTotalNumQuads(), residualFrame.getTotalQuadsSize() / BYTES_PER_MEGABYTE,
                          residualFrame.getTotalNumDepthOffsets(), residualFrame.getTotalDepthOffsetsSize() / BYTES_PER_MEGABYTE);
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
    residualFrameRT.blit(atlasVideoStreamerRT,
        0, 0, residualFrameRT.width, residualFrameRT.height,
        col, row, dstWidth, dstHeight
    );

    return renderStats;
}

void QUASARStreamer::sendProxies(pose_id_t poseID, bool createResidualFrame) {
    if (!videoURL.empty() && !proxiesURL.empty()) {
        // Send atlas frame
        atlasVideoStreamerRT.sendFrame(poseID);
        // Send proxies
        writeToMemory(poseID, createResidualFrame, compressedData);
        send(compressedData);
    }
}

size_t QUASARStreamer::writeToFiles(const Path& outputPath) {
    // Save camera data
    Pose cameraPose;
    Path cameraFileName = (outputPath / "camera").withExtension(".bin");
    cameraPose.setProjectionMatrix(remoteCamera.getProjectionMatrix());
    cameraPose.setViewMatrix(remoteCamera.getViewMatrix());
    cameraPose.writeToFile(cameraFileName);

    Path cameraFileNamePrev = (outputPath / "camera_prev").withExtension(".bin");
    cameraPose.setProjectionMatrix(remoteCameraPrev.getProjectionMatrix());
    cameraPose.setViewMatrix(remoteCameraPrev.getViewMatrix());
    cameraPose.writeToFile(cameraFileNamePrev);

    // Save metadata (viewSphereDiameter and wide FOV)
    QUASARReceiver::Params params = {
        .numLayers = static_cast<uint32_t>(geometryMetadatas.size()),
        .viewSphereDiameter = viewSphereDiameter,
        .wideFOV = remoteCameraWideFOV.getFovyDegrees(),
    };
    FileIO::writeToBinaryFile(outputPath / "metadata.bin", &params, sizeof(params));

    // Save color
    Path colorFileName = (outputPath / "color").withExtension(".jpg");
    atlasVideoStreamerRT.writeColorAsJPG(colorFileName);

    // Save proxies
    size_t totalOutputSize = 0;
    for (int layer = 0; layer < maxLayers; layer++) {
        totalOutputSize += referenceFrames[layer].writeToFiles(outputPath, layer);
    }
    totalOutputSize += residualFrame.writeToFiles(outputPath);

    spdlog::debug("Written output data size: {}", totalOutputSize);
    return totalOutputSize;
}

size_t QUASARStreamer::writeToMemory(pose_id_t poseID, bool writeResidualFrame, std::vector<char>& outputData) {
    // Save camera data
    Pose cameraPose;
    std::vector<char> cameraData;
    cameraPose.setProjectionMatrix(remoteCamera.getProjectionMatrix());
    cameraPose.setViewMatrix(remoteCamera.getViewMatrix());
    cameraPose.writeToMemory(cameraData);

    // Save geometry data
    // Save visible layer
    if (!writeResidualFrame) {
        referenceFrames[0].writeToMemory(geometryMetadatas[0]);
    }
    else {
        residualFrame.writeToMemory(geometryMetadatas[0]);
    }
    // Save hidden layers and wide FOV
    for (int layer = 1; layer < maxLayers; layer++) {
        referenceFrames[layer].writeToMemory(geometryMetadatas[layer]);
    }

    uint32_t geometrySize = 0;
    for (const auto& layerData : geometryMetadatas) {
        geometrySize += sizeof(uint32_t) + static_cast<uint32_t>(layerData.size());
    }

    QUASARReceiver::Header header{
        .poseID = poseID,
        .frameType = !writeResidualFrame ? QuadFrame::FrameType::REFERENCE : QuadFrame::FrameType::RESIDUAL,
        .cameraSize = static_cast<uint32_t>(cameraData.size()),
        .params {
            .numLayers = static_cast<uint32_t>(geometryMetadatas.size()),
            .viewSphereDiameter = viewSphereDiameter,
            .wideFOV = remoteCameraWideFOV.getFovyDegrees(),
        },
        .geometrySize = geometrySize,
    };

    spdlog::debug("Writing camera size: {}", header.cameraSize);
    spdlog::debug("Writing geometry size: {}", header.geometrySize);

    outputData.resize(sizeof(header) + cameraData.size() + geometrySize);
    char* ptr = outputData.data();

    // Write header
    std::memcpy(ptr, &header, sizeof(header));
    ptr += sizeof(header);

    // Write camera data
    std::memcpy(ptr, cameraData.data(), cameraData.size());
    ptr += cameraData.size();

    // Write geometry data
    for (const auto& layerData : geometryMetadatas) {
        uint32_t layerSize = static_cast<uint32_t>(layerData.size());

        // Write size of layer
        std::memcpy(ptr, &layerSize, sizeof(uint32_t));
        ptr += sizeof(uint32_t);

        // Write layer data
        std::memcpy(ptr, layerData.data(), layerSize);
        ptr += layerSize;
    }

    spdlog::debug("Written output data size: {}", outputData.size());
    return outputData.size();
}
