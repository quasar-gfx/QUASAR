#include <Streamers/QuadsStreamer.h>

using namespace quasar;

QuadsStreamer::QuadsStreamer(
        QuadSet& quadSet,
        DeferredRenderer& remoteRenderer,
        Scene& remoteScene,
        PerspectiveCamera& remoteCamera,
        const QuadsStreamerCreateParams& params)
    : quadSet(quadSet)
    , videoURL(params.videoURL)
    , proxiesURL(params.proxiesURL)
    , remoteRenderer(remoteRenderer)
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
    , videoAtlasStreamerRT({
        .width = 2 * quadSet.getSize().x,
        .height = quadSet.getSize().y,
        .internalFormat = GL_SRGB8_ALPHA8,
        .format = GL_RGBA,
        .type = GL_UNSIGNED_BYTE,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    }, params.videoURL, params.targetFramerate, params.targetBitRate)
    , alphaAtlasRT({
        .width = 2 * quadSet.getSize().x,
        .height = quadSet.getSize().y,
        .internalFormat = GL_R8,
        .format = GL_RED,
        .type = GL_UNSIGNED_BYTE,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    })
    , alphaCodec(alphaAtlasRT.width, alphaAtlasRT.height)
    , residualFrameMesh(quadSet, residualFrameRT_noTone.colorTexture, residualFrameRT_noTone.alphaTexture)
    , depthMesh(quadSet.getSize(), glm::vec4(0.0f, 1.0f, 0.0f, 1.0f))
    , wireframeMaterial({ .baseColor = colors[0] })
    , maskWireframeMaterial({ .baseColor = colors[colors.size()-1] })
    , DataStreamerTCP(params.proxiesURL)
{
    meshScenes.resize(2);
    referenceFrameMeshes.reserve(meshScenes.size());
    referenceFrameNodes.reserve(meshScenes.size());
    referenceFrameNodesLocal.reserve(meshScenes.size());
    referenceFrameWireframesLocal.reserve(meshScenes.size());

    remoteCameraPrev.setProjectionMatrix(remoteCamera.getProjectionMatrix());
    remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());

    for (int i = 0; i < meshScenes.size(); i++) {
        referenceFrameMeshes.emplace_back(
            quadSet, referenceFrameRT_noTone.colorTexture, referenceFrameRT_noTone.alphaTexture);

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
    residualFrameNodeLocal.addEntity(&residualFrameMesh);
    residualFrameNodeLocal.frustumCulled = false;

    residualFrameWireframesLocal.addEntity(&residualFrameMesh);
    residualFrameWireframesLocal.frustumCulled = false;
    residualFrameWireframesLocal.wireframe = true;
    residualFrameWireframesLocal.visible = false;
    residualFrameWireframesLocal.overrideMaterial = &maskWireframeMaterial;

    depthNode.addEntity(&depthMesh);
    depthNode.frustumCulled = false;
    depthNode.visible = false;
    depthNode.primitiveType = GL_POINTS;

    alphaImageData.resize(alphaAtlasRT.width * alphaAtlasRT.height);

    if (!videoURL.empty() && !proxiesURL.empty()) {
        spdlog::info("Created QuadsStreamer that sends to URL: tcp://{}", proxiesURL);
    }
}

QuadsStreamer::~QuadsStreamer() {
    videoAtlasStreamerRT.stop();
}

uint QuadsStreamer::getNumTriangles() const {
    int currMeshIndex  = meshIndex % 2;
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

RenderStats QuadsStreamer::generateFrame(bool createResidualFrame, bool showNormals, bool showDepth) {
    // Reset stats
    Stats prevStats = stats;
    stats = { 0 };
    stats.frameSize = prevStats.frameSize; // Keep previous frame size

    int currMeshIndex  = meshIndex % 2;
    int prevMeshIndex  = (meshIndex + 1) % 2;

    auto& remoteCameraToUse = createResidualFrame ? remoteCameraPrev : remoteCamera;
    auto quadsGenerator = frameGenerator.getQuadsGenerator();

    /*
    ============================
    Render scene normally to create Reference Frame textures
    ============================
    */
    double startTime = timeutils::getTimeMicros();
    RenderStats renderStats = remoteRenderer.drawObjects(remoteScene, remoteCameraToUse);
    remoteRenderer.copyToFrameRT(referenceFrameRT);
    stats.totalRenderTimeMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    /*
    ============================
    Generate Reference Frame
    ============================
    */
    quadsGenerator->params.expandEdges = false;
    ReferenceFrame dummyFrame;
    frameGenerator.createReferenceFrame(
        referenceFrameRT,
        remoteCameraToUse,
        referenceFrameMeshes[currMeshIndex],
        createResidualFrame ? dummyFrame : referenceFrame // Don't save output of this reference frame if we are making a residual frame
    );
    if (!showNormals) {
        remoteRenderer.copyToFrameRT(referenceFrameRT_noTone);
        tonemapper.drawToRenderTarget(remoteRenderer, referenceFrameRT);
    }
    else {
        showNormalsEffect.drawToRenderTarget(remoteRenderer, referenceFrameRT_noTone);
    }

    stats.totalGenQuadMapTimeMs += frameGenerator.stats.generateQuadsTimeMs;
    stats.totalSimplifyTimeMs += frameGenerator.stats.simplifyQuadsTimeMs;
    stats.totalGatherQuadsTime += frameGenerator.stats.gatherQuadsTimeMs;
    stats.totalCreateProxiesTimeMs += frameGenerator.stats.createQuadsTimeMs;

    stats.totalAppendQuadsTimeMs += frameGenerator.stats.appendQuadsTimeMs;
    stats.totalCreateVertIndTimeMs += frameGenerator.stats.createVertIndTimeMs;
    stats.totalCreateMeshTimeMs += frameGenerator.stats.createMeshTimeMs;

    stats.totalCompressTimeMs += frameGenerator.stats.compressTimeMs;

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

    residualFrameNodeLocal.visible = createResidualFrame;

    // Update color atlas texture
    referenceFrameRT.blit(videoAtlasStreamerRT,
        0, 0, referenceFrameRT.width, referenceFrameRT.height,
        0, 0, referenceFrameRT.width, referenceFrameRT.height
    );
    residualFrameRT.blit(videoAtlasStreamerRT,
        0, 0, residualFrameRT.width, residualFrameRT.height,
        referenceFrameRT.width, 0, videoAtlasStreamerRT.width, videoAtlasStreamerRT.height
    );

    // Update alpha atlas texture
    referenceFrameRT.blit(alphaAtlasRT,
        0, 0, referenceFrameRT.width, referenceFrameRT.height,
        0, 0, referenceFrameRT.width, referenceFrameRT.height
    );
    residualFrameRT_noTone.blit(alphaAtlasRT,
        0, 0, residualFrameRT_noTone.width, residualFrameRT_noTone.height,
        referenceFrameRT.width, 0, alphaAtlasRT.width, alphaAtlasRT.height
    );

    // For debugging: Generate point cloud from depth map
    if (showDepth) {
        depthMesh.update(remoteCamera, referenceFrameRT);
        stats.totalGenDepthTimeMs += depthMesh.stats.genDepthTime;
    }

    if (!createResidualFrame) {
        stats.proxySizes.numQuads += referenceFrame.getTotalNumQuads();
        stats.proxySizes.numDepthOffsets += referenceFrame.getTotalNumDepthOffsets();
        stats.proxySizes.quadsSize += referenceFrame.getTotalQuadsSize();
        stats.proxySizes.depthOffsetsSize += referenceFrame.getTotalDepthOffsetsSize();
        spdlog::debug("Reference frame generated with {} quads ({:.3f}MB), {} depth offsets ({:.3f}MB)",
                      referenceFrame.getTotalNumQuads(), referenceFrame.getTotalQuadsSize() / BYTES_PER_MEGABYTE,
                      referenceFrame.getTotalNumDepthOffsets(), referenceFrame.getTotalDepthOffsetsSize() / BYTES_PER_MEGABYTE);
    }
    else {
        stats.proxySizes.numQuads += residualFrame.getTotalNumQuads();
        stats.proxySizes.numDepthOffsets += residualFrame.getTotalNumDepthOffsets();
        stats.proxySizes.quadsSize += residualFrame.getTotalQuadsSize();
        stats.proxySizes.depthOffsetsSize += residualFrame.getTotalDepthOffsetsSize();
        spdlog::debug("Residual frame generated with {} updated quads ({:.3f}MB) and {} revealed quads ({:.3f}MB), {} updated depth offsets ({:.3f}MB) and {} revealed depth offsets ({:.3f}MB)",
                      residualFrame.getTotalNumQuadsUpdated(), residualFrame.getTotalQuadsUpdatedSize() / BYTES_PER_MEGABYTE,
                      residualFrame.getTotalNumQuadsRevealed(), residualFrame.getTotalQuadsRevealedSize() / BYTES_PER_MEGABYTE,
                      residualFrame.getTotalNumDepthOffsetsUpdated(), residualFrame.getTotalDepthOffsetsUpdatedSize() / BYTES_PER_MEGABYTE,
                      residualFrame.getTotalNumDepthOffsetsRevealed(), residualFrame.getTotalDepthOffsetsRevealedSize() / BYTES_PER_MEGABYTE);
    }

    return renderStats;
}

void QuadsStreamer::sendFrame(pose_id_t poseID, bool createResidualFrame) {
    stats.frameSize = writeToMemory(poseID, createResidualFrame, compressedData);
    if (!videoURL.empty() && !proxiesURL.empty()) {
        // Send atlas frame
        videoAtlasStreamerRT.sendFrame(poseID);
        // Send proxies
        send(compressedData);
    }
}

void QuadsStreamer::writeTexturesToFiles(const Path& outputPath) {
    // Save color
    Path colorFileName = (outputPath / "color.jpg");
    videoAtlasStreamerRT.writeColorAsJPG(colorFileName);

    // Save alpha
    Path alphaFileName = (outputPath / "alpha.png");
    alphaAtlasRT.writeAlphaAsPNG(alphaFileName);
}

size_t QuadsStreamer::writeToFiles(const Path& outputPath) {
    // Save camera data
    Pose cameraPose;
    Path cameraFileName = outputPath / "camera.bin";
    cameraPose.setProjectionMatrix(remoteCamera.getProjectionMatrix());
    cameraPose.setViewMatrix(remoteCamera.getViewMatrix());
    cameraPose.writeToFile(cameraFileName);

    Path cameraFileNamePrev = outputPath / "camera_prev.bin";
    cameraPose.setProjectionMatrix(remoteCameraPrev.getProjectionMatrix());
    cameraPose.setViewMatrix(remoteCameraPrev.getViewMatrix());
    cameraPose.writeToFile(cameraFileNamePrev);

    writeTexturesToFiles(outputPath);

    // Save proxies
    size_t totalOutputSize = referenceFrame.writeToFiles(outputPath) + residualFrame.writeToFiles(outputPath);
    return totalOutputSize;
}

size_t QuadsStreamer::writeToMemory(pose_id_t poseID, bool writeResidualFrame, std::vector<char>& outputData) {
    // Save camera data
    Pose cameraPose;
    cameraPose.setProjectionMatrix(remoteCamera.getProjectionMatrix());
    cameraPose.setViewMatrix(remoteCamera.getViewMatrix());
    cameraPose.writeToMemory(cameraData);

    // Save alpha atlas
    alphaAtlasRT.writeAlphaToMemory(alphaImageData);
    alphaCodec.compress(alphaImageData.data(), alphaData, alphaImageData.size());

    // Save geometry data
    if (!writeResidualFrame) {
        referenceFrame.writeToMemory(geometryData);
    }
    else {
        residualFrame.writeToMemory(geometryData);
    }

    QuadsReceiver::Header header{
        .poseID = poseID,
        .frameType = !writeResidualFrame ? QuadFrame::FrameType::REFERENCE : QuadFrame::FrameType::RESIDUAL,
        .cameraSize = static_cast<uint32_t>(cameraData.size()),
        .alphaSize = static_cast<uint32_t>(alphaData.size()),
        .geometrySize = static_cast<uint32_t>(geometryData.size())
    };

    spdlog::debug("Writing camera size: {:.3f}MB", static_cast<float>(header.cameraSize) / BYTES_PER_MEGABYTE);
    spdlog::debug("Writing alpha size: {:.3f}MB", static_cast<float>(header.alphaSize) / BYTES_PER_MEGABYTE);
    spdlog::debug("Writing geometry size: {:.3f}MB", static_cast<float>(header.geometrySize) / BYTES_PER_MEGABYTE);

    outputData.resize(header.getSize());
    char* ptr = outputData.data();

    // Write header
    std::memcpy(ptr, &header, sizeof(header));
    ptr += sizeof(header);

    // Write camera data
    std::memcpy(ptr, cameraData.data(), cameraData.size());
    ptr += cameraData.size();

    // Write alpha data
    std::memcpy(ptr, alphaData.data(), alphaData.size());
    ptr += alphaData.size();

    // Write geometry data
    std::memcpy(ptr, geometryData.data(), geometryData.size());
    ptr += geometryData.size();

    spdlog::debug("Total data size: {:.3f}MB", static_cast<float>(outputData.size()) / BYTES_PER_MEGABYTE);

    return outputData.size();
}
