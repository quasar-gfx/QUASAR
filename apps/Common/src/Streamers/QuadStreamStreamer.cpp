#include <Streamers/QuadStreamStreamer.h>

using namespace quasar;

QuadStreamStreamer::QuadStreamStreamer(
        QuadSet& quadSet,
        DeferredRenderer& remoteRenderer,
        Scene& remoteScene,
        PerspectiveCamera& remoteCamera,
        const QuadStreamStreamerCreateParams& params)
    : quadSet(quadSet)
    , maxViews(params.maxViews)
    , remoteRenderer(remoteRenderer)
    , remoteScene(remoteScene)
    , remoteCamera(remoteCamera)
    , frameGenerator(quadSet)
    , wireframeMaterial({ .baseColor = colors[0] })
    , maskWireframeMaterial({ .baseColor = colors[colors.size()-1] })
{
    referenceFrameRTs.reserve(maxViews);
    referenceFrameRTs_noTone.reserve(maxViews);
    referenceFrameMeshes.reserve(maxViews);
    depthMeshes.reserve(maxViews);
    referenceFrameNodesLocal.reserve(maxViews);
    referenceFrameNodesRemote.reserve(maxViews);
    referenceFrameWireframesLocal.reserve(maxViews);
    depthNodes.reserve(maxViews);

    remoteCameras.resize(maxViews);
    referenceFrames.resize(maxViews);

    // Match QuadStream's params from paper:
    auto quadsGenerator = frameGenerator.getQuadsGenerator();
    quadsGenerator->params.expandProxies = false;
    quadsGenerator->params.expandEdges = true;
    quadsGenerator->params.depthThreshold = 1e-4f;
    quadsGenerator->params.planeSimilarityThreshold = 1e-2f;
    quadsGenerator->params.flattenThreshold = 1e-1f;
    quadsGenerator->params.maxIterForceMerge = 1; // Only merge once (similar-ish to doing quad splitting)
    frameGenerator.params.applyDeltaEncoding = false;

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
    for (int view = 0; view < maxViews; view++) {
        if (view == maxViews - 1) {
            rtParams.width = 1280; rtParams.height = 720;
        }

        remoteCameras[view].setProjectionMatrix(remoteCameras[view].getProjectionMatrix());
        if (view == maxViews - 1) {
            remoteCameras[view].setFovyDegrees(params.wideFOV);
        }
        else {
            remoteCameras[view].setFovyDegrees(remoteCamera.getFovyDegrees());
        }
        remoteCameras[view].setViewMatrix(remoteCamera.getViewMatrix());

        referenceFrameRTs.emplace_back(rtParams);
        referenceFrameRTs_noTone.emplace_back(rtParams);

        referenceFrameMeshes.emplace_back(quadSet, referenceFrameRTs_noTone[view].colorTexture, referenceFrameRTs_noTone[view].alphaTexture);
        referenceFrameMeshes[view].setExpandQuadAmount(0.0f);
        referenceFrameNodesLocal.emplace_back(&referenceFrameMeshes[view]);
        referenceFrameNodesLocal[view].frustumCulled = false;

        const glm::vec4& color = colors[view % colors.size()];
        referenceFrameWireframesLocal.emplace_back(&referenceFrameMeshes[view]);
        referenceFrameWireframesLocal[view].frustumCulled = false;
        referenceFrameWireframesLocal[view].wireframe = true;
        referenceFrameWireframesLocal[view].overrideMaterial = new QuadMaterial({ .baseColor = color });

        depthMeshes.emplace_back(quadSet.getSize(), glm::vec4(0.0f, 1.0f, 0.0f, 1.0f));
        depthNodes.emplace_back(&depthMeshes[view]);
        depthNodes[view].frustumCulled = false;
        depthNodes[view].primitiveType = GL_POINTS;

        referenceFrameNodesRemote.emplace_back(&referenceFrameMeshes[view]);
        referenceFrameNodesRemote[view].frustumCulled = false;
        referenceFrameNodesRemote[view].visible = (view == 0); // Center view is always visible
        meshScene.addChildNode(&referenceFrameNodesRemote[view]);
    }

    setViewBoxSize(params.viewBoxSize);
}

uint QuadStreamStreamer::getNumTriangles() const {
    uint numTriangles = 0;
    for (const auto& mesh : referenceFrameMeshes) {
        auto size = mesh.getBufferSizes();
        numTriangles += size.numIndices / 3; // Each triangle has 3 indices
    }
    return numTriangles;
}

void QuadStreamStreamer::addMeshesToScene(Scene& localScene) {
    // Add in reverse order to have correct layering
    for (int view = maxViews - 1; view >= 0; view--) {
        localScene.addChildNode(&referenceFrameNodesLocal[view]);
        localScene.addChildNode(&referenceFrameWireframesLocal[view]);
        localScene.addChildNode(&depthNodes[view]);
    }
}

void QuadStreamStreamer::setViewBoxSize(float viewBoxSize) {
    this->viewBoxSize = viewBoxSize;
}

RenderStats QuadStreamStreamer::generateFrame(bool showNormals, bool showDepth) {
    // Reset stats
    stats = { 0 };
    RenderStats renderStats;

    PerspectiveCamera& remoteCameraCenter = remoteCameras[0];
    remoteCameraCenter.setViewMatrix(remoteCamera.getViewMatrix());
    remoteCameraCenter.setPosition(remoteCamera.getPosition());

    // Update other cameras in view box corners
    for (int view = 1; view < maxViews - 1; view++) {
        const glm::vec3& offset = offsets[view - 1];
        const glm::vec3& right = remoteCameraCenter.getRightVector();
        const glm::vec3& up = remoteCameraCenter.getUpVector();
        const glm::vec3& forward = remoteCameraCenter.getForwardVector();

        glm::vec3 worldOffset =
            right   * +offset.x * viewBoxSize / 2.0f +
            up      * +offset.y * viewBoxSize / 2.0f +
            forward * -offset.z * viewBoxSize / 2.0f;

        remoteCameras[view].setViewMatrix(remoteCameraCenter.getViewMatrix());
        remoteCameras[view].setPosition(remoteCameraCenter.getPosition() + worldOffset);
        remoteCameras[view].updateViewMatrix();
    }

    // Update wide fov camera
    remoteCameras[maxViews-1].setViewMatrix(remoteCameraCenter.getViewMatrix());

    for (int view = 0; view < maxViews; view++) {
        auto& remoteCameraToUse = remoteCameras[view];
        auto& renderTargetToUse = referenceFrameRTs[view];
        auto& renderTargetToUse_noTone = referenceFrameRTs_noTone[view];
        auto& meshToUse = referenceFrameMeshes[view];
        auto& depthMeshToUse = depthMeshes[view];

        double startTime = timeutils::getTimeMicros();

        // Center view
        if (view == 0) {
            // Render all objects in remoteScene normally
            renderStats = remoteRenderer.drawObjects(remoteScene, remoteCameraToUse);
        }
        // Other view
        else {
            // Make all previous referenceFrameMeshes visible and everything else invisible
            for (int prevView = 1; prevView < maxViews; prevView++) {
                meshScene.children[prevView]->visible = (prevView < view);
            }
            // Draw old referenceFrameMeshes at new remoteCamera view, filling stencil buffer with 1
            remoteRenderer.pipeline.stencilState.enableRenderingIntoStencilBuffer(GL_KEEP, GL_KEEP, GL_REPLACE);
            remoteRenderer.pipeline.writeMaskState.disableColorWrites();
            renderStats += remoteRenderer.drawObjectsNoLighting(meshScene, remoteCameraToUse);

            // Render remoteScene using stencil buffer as a mask
            // At values where stencil buffer is not 1, remoteScene should render
            remoteRenderer.pipeline.stencilState.enableRenderingUsingStencilBufferAsMask(GL_NOTEQUAL, 1);
            remoteRenderer.pipeline.rasterState.polygonOffsetEnabled = false;
            remoteRenderer.pipeline.writeMaskState.enableColorWrites();
            renderStats += remoteRenderer.drawObjectsNoLighting(remoteScene, remoteCameraToUse, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            remoteRenderer.pipeline.stencilState.restoreStencilState();
        }
        remoteRenderer.copyToFrameRT(renderTargetToUse);
        stats.totalRenderTimeMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        /*
        ============================
        Generate Reference Frame
        ============================
        */
        frameGenerator.createReferenceFrame(
            renderTargetToUse, remoteCameraToUse,
            meshToUse,
            referenceFrames[view]
        );
        if (!showNormals) {
            renderTargetToUse.blit(renderTargetToUse_noTone);
            tonemapper.setUniforms(renderTargetToUse_noTone);
            tonemapper.drawToRenderTarget(remoteRenderer, renderTargetToUse, false);
        }
        else {
            showNormalsEffect.drawToRenderTarget(remoteRenderer, renderTargetToUse_noTone);
        }

        stats.totalGenQuadMapTimeMs += frameGenerator.stats.generateQuadsTimeMs;
        stats.totalSimplifyTimeMs += frameGenerator.stats.simplifyQuadsTimeMs;
        stats.totalGatherQuadsTime += frameGenerator.stats.gatherQuadsTimeMs;
        stats.totalCreateProxiesTimeMs += frameGenerator.stats.createQuadsTimeMs;

        stats.totalAppendQuadsTimeMs += frameGenerator.stats.appendQuadsTimeMs;
        stats.totalCreateVertIndTimeMs += frameGenerator.stats.createVertIndTimeMs;
        stats.totalCreateMeshTimeMs += frameGenerator.stats.createMeshTimeMs;

        stats.totalCompressTimeMs += frameGenerator.stats.compressTimeMs;

        // For debugging: Generate point cloud from depth map
        if (showDepth) {
            depthMeshToUse.update(remoteCameraToUse, renderTargetToUse);
            stats.totalGenDepthTimeMs += depthMeshToUse.stats.genDepthTime;
        }

        stats.proxySizes.numQuads += referenceFrames[view].getTotalNumQuads();
        stats.proxySizes.numDepthOffsets += referenceFrames[view].getTotalNumDepthOffsets();
        // QS has data structures that are 103 bits
        // We approximate their data size by multiplying by 103/sizeof(our quad data struct)
        stats.proxySizes.quadsSize += referenceFrames[view].getTotalQuadsSize() * (103.0 / (8 * sizeof(QuadMapDataPacked)));
        stats.proxySizes.depthOffsetsSize += referenceFrames[view].getTotalDepthOffsetsSize();
        spdlog::debug("Reference frame generated with {} quads ({:.3f}MB), {} depth offsets ({:.3f}MB)",
            referenceFrames[view].getTotalNumQuads(), referenceFrames[view].getTotalQuadsSize() * (103.0 / (8 * sizeof(QuadMapDataPacked))) / BYTES_PER_MEGABYTE,
            referenceFrames[view].getTotalNumDepthOffsets(), referenceFrames[view].getTotalDepthOffsetsSize() / BYTES_PER_MEGABYTE);
    }

    stats.frameSize = stats.proxySizes.quadsSize + stats.proxySizes.depthOffsetsSize;
    return renderStats;
}

size_t QuadStreamStreamer::writeToFiles(const Path& outputPath) {
    // Save camera data
    Pose cameraPose;
    PerspectiveCamera& remoteCameraCenter = remoteCameras[0];
    Path cameraFileName = outputPath / "camera.bin";
    cameraPose.setProjectionMatrix(remoteCameraCenter.getProjectionMatrix());
    cameraPose.setViewMatrix(remoteCameraCenter.getViewMatrix());
    cameraPose.writeToFile(cameraFileName);

    // Save metadata (viewBoxSize and wide FOV)
    PerspectiveCamera& remoteCameraWideFOV = remoteCameras[maxViews-1];
    std::vector<float> metadata = {
        remoteCameraWideFOV.getFovyDegrees(),
        viewBoxSize,
    };
    FileIO::writeToBinaryFile(outputPath / "metadata.bin", metadata.data(), metadata.size() * sizeof(float));

    // Save color + alpha data and proxies
    size_t totalOutputSize = 0;
    for (int view = 0; view < maxViews; view++) {
        Path colorFileName = outputPath / ("color" + std::to_string(view) + ".jpg");
        referenceFrameRTs[view].writeColorAsJPG(colorFileName);

        Path alphaFileName = (outputPath / ("alpha" + std::to_string(view))).withExtension(".png");
        referenceFrameRTs[view].writeAlphaAsPNG(alphaFileName);

        totalOutputSize += referenceFrames[view].writeToFiles(outputPath, view);
    }
    return totalOutputSize;
}
