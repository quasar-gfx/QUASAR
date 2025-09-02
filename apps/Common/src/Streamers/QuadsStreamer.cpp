#include <Streamers/QuadsStreamer.h>

using namespace quasar;

QuadsStreamer::QuadsStreamer(
        QuadSet& quadSet,
        DeferredRenderer& remoteRenderer, Scene& remoteScene,
        const std::string& videoURL,
        const std::string& proxiesURL)
    : videoURL(videoURL)
    , proxiesURL(proxiesURL)
    , quadSet(quadSet)
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
    , atlasVideoStreamerRT({
        .width = 2 * quadSet.getSize().x,
        .height = quadSet.getSize().y,
        .internalFormat = GL_SRGB8,
        .format = GL_RGB,
        .type = GL_UNSIGNED_BYTE,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    }, videoURL)
    // We can use less vertices and indicies for the mask since it will be sparser
    , residualFrameMesh(quadSet, residualFrameRT_noTone.colorTexture, MAX_QUADS_PER_MESH / 4)
    , depthMesh(quadSet.getSize(), glm::vec4(0.0f, 1.0f, 0.0f, 1.0f))
    , wireframeMaterial({ .baseColor = colors[0] })
    , maskWireframeMaterial({ .baseColor = colors[colors.size()-1] })
    , DataStreamerTCP(proxiesURL)
{
    meshScenes.resize(2);
    referenceFrameMeshes.reserve(meshScenes.size());
    referenceFrameNodes.reserve(meshScenes.size());
    referenceFrameNodesLocal.reserve(meshScenes.size());
    referenceFrameWireframesLocal.reserve(meshScenes.size());

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
    residualFrameNodeLocal.setEntity(&residualFrameMesh);
    residualFrameNodeLocal.frustumCulled = false;

    residualFrameWireframesLocal.setEntity(&residualFrameMesh);
    residualFrameWireframesLocal.frustumCulled = false;
    residualFrameWireframesLocal.wireframe = true;
    residualFrameWireframesLocal.visible = false;
    residualFrameWireframesLocal.overrideMaterial = &maskWireframeMaterial;

    depthNode.setEntity(&depthMesh);
    depthNode.frustumCulled = false;
    depthNode.visible = false;
    depthNode.primitiveType = GL_POINTS;

    if (!videoURL.empty() && !proxiesURL.empty()) {
        spdlog::info("Created QuadsStreamer that sends to URL: {}", proxiesURL);
    }
}

QuadsStreamer::~QuadsStreamer() {
    atlasVideoStreamerRT.stop();
}

uint QuadsStreamer::getNumTriangles() const {
    auto refMeshSizes = referenceFrameMeshes[currMeshIndex].getBufferSizes();
    auto maskMeshSizes = residualFrameMesh.getBufferSizes();
    return (refMeshSizes.numIndices + maskMeshSizes.numIndices) / 3; // Each triangle has 3 indices
}

void QuadsStreamer::addMeshesToScene(Scene& localScene) {
    for (int i = 0; i < 2; i++) {
        localScene.addChildNode(&referenceFrameNodesLocal[i]);
        localScene.addChildNode(&referenceFrameWireframesLocal[i]);
    }
    localScene.addChildNode(&residualFrameNodeLocal);
    localScene.addChildNode(&residualFrameWireframesLocal);
    localScene.addChildNode(&depthNode);
}

void QuadsStreamer::generateFrame(
    DeferredRenderer& remoteRenderer, Scene& remoteScene, const PerspectiveCamera& remoteCamera,
    bool createResidualFrame,
    bool showNormals, bool showDepth)
{
    // Reset stats
    stats = { 0 };

    auto& remoteCameraToUse = createResidualFrame ? remoteCameraPrev : remoteCamera;
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
    remoteRenderer.drawObjects(remoteScene, remoteCameraToUse);
    remoteRenderer.copyToFrameRT(referenceFrameRT);
    stats.totalRenderTime += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    /*
    ============================
    Generate Reference Frame
    ============================
    */
    quadsGenerator->params.expandEdges = false;
    ReferenceFrame currReferenceFrame;
    frameGenerator.createReferenceFrame(
        referenceFrameRT,
        remoteCameraToUse,
        referenceFrameMeshes[currMeshIndex],
        createResidualFrame ? currReferenceFrame : referenceFrame
    );
    if (!showNormals) {
        remoteRenderer.copyToFrameRT(referenceFrameRT_noTone);
        toneMapper.drawToRenderTarget(remoteRenderer, referenceFrameRT);
    }
    else {
        showNormalsEffect.drawToRenderTarget(remoteRenderer, referenceFrameRT_noTone);
    }

    stats.totalGenQuadMapTime += frameGenerator.stats.timeToGenerateQuadsMs;
    stats.totalSimplifyTime += frameGenerator.stats.timeToSimplifyQuadsMs;
    stats.totalGatherQuadsTime += frameGenerator.stats.timeToGatherQuadsMs;
    stats.totalCreateProxiesTime += frameGenerator.stats.timeToCreateQuadsMs;

    stats.totalAppendQuadsTime += frameGenerator.stats.timeToAppendQuadsMs;
    stats.totalFillQuadsIndiciesTime += frameGenerator.stats.timeToFillQuadIndicesMs;
    stats.totalCreateVertIndTime += frameGenerator.stats.timeToCreateVertIndMs;
    stats.totaltimeToCreateMeshMs += frameGenerator.stats.timeToCreateMeshMs;

    stats.totalCompressTime += frameGenerator.stats.timeToCompressMs;

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
    else {
        // Only update the previous camera pose if we are not generating a Residual Frame
        remoteCameraPrev.setProjectionMatrix(remoteCamera.getProjectionMatrix());
        remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());
    }

    residualFrameNodeLocal.visible = createResidualFrame;

    // Update atlas texture
    referenceFrameRT.blit(atlasVideoStreamerRT,
        0, 0, referenceFrameRT.width, referenceFrameRT.height,
        0, 0, referenceFrameRT.width, referenceFrameRT.height
    );
    if (createResidualFrame) {
        residualFrameRT.blit(atlasVideoStreamerRT,
            0, 0, residualFrameRT.width, residualFrameRT.height,
            referenceFrameRT.width, 0, atlasVideoStreamerRT.width, atlasVideoStreamerRT.height
        );
    }

    // For debugging: Generate point cloud from depth map
    if (showDepth) {
        depthMesh.update(remoteCamera, referenceFrameRT);
        stats.totalGenDepthTime += depthMesh.stats.genDepthTime;
    }

    if (!createResidualFrame) {
        stats.totalSizes.numQuads += referenceFrame.getTotalNumQuads();
        stats.totalSizes.numDepthOffsets += referenceFrame.getTotalNumDepthOffsets();
        stats.totalSizes.quadsSize += referenceFrame.getTotalQuadsSize();
        stats.totalSizes.depthOffsetsSize += referenceFrame.getTotalDepthOffsetsSize();
        spdlog::debug("Reference frame generated with {} quads, {} depth offsets",
                     referenceFrame.getTotalNumQuads(), referenceFrame.getTotalNumDepthOffsets());
    }
    else {
        stats.totalSizes.numQuads += residualFrame.getTotalNumQuads();
        stats.totalSizes.numDepthOffsets += residualFrame.getTotalNumDepthOffsets();
        stats.totalSizes.quadsSize += residualFrame.getTotalQuadsSize();
        stats.totalSizes.depthOffsetsSize += residualFrame.getTotalDepthOffsetsSize();
        spdlog::debug("Residual frame generated with {} quads ({} revealed), {} depth offsets",
                      residualFrame.getTotalNumQuads(),
                      residualFrame.getTotalNumQuadsRevealed(), residualFrame.getTotalNumDepthOffsets());
    }
}

void QuadsStreamer::sendProxies(pose_id_t poseID, const PerspectiveCamera& remoteCamera, bool createResidualFrame) {
    if (!videoURL.empty() && !proxiesURL.empty()) {
        // Send atlas frame
        atlasVideoStreamerRT.sendFrame(poseID);
        // Send proxies
        writeToMemory(poseID, remoteCamera, createResidualFrame, compressedData);
        send(compressedData);
    }
}

size_t QuadsStreamer::writeToFiles(const PerspectiveCamera& remoteCamera, const Path& outputPath) {
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

    // Save color
    Path colorFileName = (outputPath / "color").withExtension(".jpg");
    atlasVideoStreamerRT.writeColorAsJPG(colorFileName);

    // Save proxies
    return referenceFrame.writeToFiles(outputPath) +
           residualFrame.writeToFiles(outputPath);
}

size_t QuadsStreamer::writeToMemory(
    pose_id_t poseID, const PerspectiveCamera& remoteCamera,
    bool isResidualFrame,
    std::vector<char>& outputData)
{
    // Save camera data
    Pose cameraPose;
    std::vector<char> cameraData;
    cameraPose.setProjectionMatrix(remoteCamera.getProjectionMatrix());
    cameraPose.setViewMatrix(remoteCamera.getViewMatrix());
    cameraPose.writeToMemory(cameraData);

    // Save geometry data
    if (!isResidualFrame) {
        referenceFrame.writeToMemory(geometryData);
    }
    else {
        residualFrame.writeToMemory(geometryData);
    }

    QuadsReceiver::Header header{
        .poseID = poseID,
        .frameType = !isResidualFrame ? FrameType::REFERENCE : FrameType::RESIDUAL,
        .cameraSize = static_cast<uint32_t>(cameraData.size()),
        .geometrySize = static_cast<uint32_t>(geometryData.size())
    };

    spdlog::debug("Writing camera size: {}", header.cameraSize);
    spdlog::debug("Writing geometry size: {}", header.geometrySize);

    outputData.resize(sizeof(header) + cameraData.size() + geometryData.size());

    char* ptr = outputData.data();
    // Write header
    std::memcpy(ptr, &header, sizeof(header));
    ptr += sizeof(header);
    // Write camera data
    std::memcpy(ptr, cameraData.data(), cameraData.size());
    ptr += cameraData.size();
    // Write geometry data
    std::memcpy(ptr, geometryData.data(), geometryData.size());
    ptr += geometryData.size();

    return outputData.size();
}
