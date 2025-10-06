#include <future>
#include <Quads/FrameGenerator.h>

using namespace quasar;

FrameGenerator::FrameGenerator(QuadSet& quadSet)
    : quadSet(quadSet)
{
    quadsGenerator = std::make_shared<QuadsGenerator>(quadSet);
    threadPool = std::make_unique<BS::thread_pool<>>(4);
}

void FrameGenerator::createReferenceFrame(
    const FrameRenderTarget& referenceFrameRT, const PerspectiveCamera& remoteCamera,
    QuadMesh& referenceMesh,
    ReferenceFrame& referenceFrame)
{
    stats = { 0 };

    const glm::vec2 gBufferSize = glm::vec2(referenceFrameRT.width, referenceFrameRT.height);

    /*
    ============================
    Create proxies from the current FrameRenderTarget (which includes depth and normals)
    ============================
    */
    double startTime = timeutils::getTimeMicros();
    quadsGenerator->createProxiesFromRT(referenceFrameRT, remoteCamera);
    stats.generateQuadsTimeMs = quadsGenerator->stats.generateQuadsTimeMs;
    stats.simplifyQuadsTimeMs = quadsGenerator->stats.simplifyQuadsTimeMs;
    stats.gatherQuadsTimeMs = quadsGenerator->stats.gatherQuadsTimeMs;
    stats.createQuadsTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    // Transfer updated proxies to CPU for compression
    auto sizes = quadSet.writeToMemory(uncompressedQuads, uncompressedOffsets, params.applyDeltaEncoding);
    referenceFrame.numQuads = sizes.numQuads;
    referenceFrame.numDepthOffsets = sizes.numDepthOffsets;
    stats.transferTimeMs = quadSet.stats.transferTimeMs;

    // Compress proxies (nonblocking)
    auto offsetsFuture = threadPool->submit_task([&]() {
        return referenceFrame.compressAndStoreDepthOffsets(uncompressedOffsets);
    });
    auto quadsFuture = threadPool->submit_task([&]() {
        return referenceFrame.compressAndStoreQuads(uncompressedQuads);
    });

    // Using GPU buffers, reconstruct mesh using proxies
    startTime = timeutils::getTimeMicros();
    referenceMesh.appendQuads(quadSet, gBufferSize);
    referenceMesh.createMeshFromProxies(quadSet, gBufferSize, remoteCamera);
    stats.appendQuadsTimeMs = referenceMesh.stats.appendQuadsTimeMs;
    stats.createVertIndTimeMs = referenceMesh.stats.createMeshTimeMs;
    stats.createMeshTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    /*
    ============================
    Wait for asynchronous compression to finish and set resulting data sizes
    ============================
    */
    referenceFrame.quads.resize(quadsFuture.get());
    referenceFrame.depthOffsets.resize(offsetsFuture.get());
    stats.compressTimeMs = referenceFrame.getCompressTime();
}

void FrameGenerator::updateResidualRenderTargets(
    FrameRenderTarget& residualFrameMaskRT, FrameRenderTarget& residualFrameRT,
    DeferredRenderer& remoteRenderer, Scene& remoteScene,
    Scene& currMeshScene, Scene& prevMeshScene,
    const PerspectiveCamera& currRemoteCamera, const PerspectiveCamera& remoteCameraPrev)
{
    stats.updateRTsTimeMs = 0.0;

    /*
    ============================
    Generate frame from old camera pose using previous frame as a mask to capture scene changes
    ============================
    */
    double startTime = timeutils::getTimeMicros();

    // Fill depth buffer with previous reconstructed mesh
    remoteRenderer.pipeline.writeMaskState.disableColorWrites();
    remoteRenderer.drawObjectsNoLighting(prevMeshScene, remoteCameraPrev);

    // Use current reconstructed mesh as a stencil mask
    remoteRenderer.pipeline.stencilState.enableRenderingIntoStencilBuffer(GL_KEEP, GL_KEEP, GL_REPLACE);
    remoteRenderer.pipeline.depthState.depthFunc = GL_EQUAL;
    remoteRenderer.drawObjectsNoLighting(currMeshScene, remoteCameraPrev, GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    // Render scene using stencil mask; this lets only content that is different pass
    remoteRenderer.pipeline.stencilState.enableRenderingUsingStencilBufferAsMask(GL_NOTEQUAL, 1);
    remoteRenderer.pipeline.depthState.depthFunc = GL_LESS;
    remoteRenderer.pipeline.writeMaskState.enableColorWrites();
    remoteRenderer.drawObjects(remoteScene, remoteCameraPrev, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    remoteRenderer.pipeline.stencilState.restoreStencilState();
    remoteRenderer.copyToFrameRT(residualFrameMaskRT); // Save result into a temporary render target

    /*
    ============================
    Generate frame from new camera pose using current frame as a mask to capture disocclusions due to camera movement
    ============================
    */
    // Use current reconstructed mesh as a stencil mask
    remoteRenderer.pipeline.stencilState.enableRenderingIntoStencilBuffer(GL_KEEP, GL_KEEP, GL_REPLACE);
    remoteRenderer.pipeline.writeMaskState.disableColorWrites();
    remoteRenderer.drawObjectsNoLighting(currMeshScene, currRemoteCamera);

    // Render scene using stencil mask; this lets only content that is different pass
    remoteRenderer.pipeline.stencilState.enableRenderingUsingStencilBufferAsMask(GL_NOTEQUAL, 1);
    remoteRenderer.pipeline.writeMaskState.enableColorWrites();
    remoteRenderer.drawObjects(remoteScene, currRemoteCamera, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    remoteRenderer.pipeline.stencilState.restoreStencilState();
    remoteRenderer.copyToFrameRT(residualFrameRT);

    stats.updateRTsTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
}

void FrameGenerator::createResidualFrame(
    const FrameRenderTarget& residualFrameMaskRT, const FrameRenderTarget& residualFrameRT,
    const PerspectiveCamera& currRemoteCamera, const PerspectiveCamera& remoteCameraPrev,
    QuadMesh& referenceMesh, QuadMesh& residualMesh,
    ResidualFrame& residualFrame)
{
    double updateRTsTimeMs = stats.updateRTsTimeMs;
    stats = { 0 };
    stats.updateRTsTimeMs = updateRTsTimeMs;

    const glm::vec2 gBufferSize = glm::vec2(residualFrameRT.width, residualFrameRT.height);

    /*
    ============================
    Create proxies and meshes for the masked portion of the residual frame capturing updated geometry
    ============================
    */
    double startTime = timeutils::getTimeMicros();
    quadsGenerator->createProxiesFromRT(residualFrameMaskRT, remoteCameraPrev);
    stats.createQuadsTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    // Transfer updated proxies to CPU for compression
    auto sizesUpdated = quadSet.writeToMemory(uncompressedQuads, uncompressedOffsets, params.applyDeltaEncoding);
    residualFrame.numQuadsUpdated = sizesUpdated.numQuads;
    residualFrame.numDepthOffsetsUpdated = sizesUpdated.numDepthOffsets;
    stats.transferTimeMs = quadSet.stats.transferTimeMs;

    // Compress updated proxies (asynchronous)
    auto offsetsUpdatedFuture = threadPool->submit_task([&]() {
        return residualFrame.compressAndStoreUpdatedDepthOffsets(uncompressedOffsets);
    });
    auto quadsUpdatedFuture = threadPool->submit_task([&]() {
        return residualFrame.compressAndStoreUpdatedQuads(uncompressedQuads);
    });

    // Using GPU buffers, update reference frame mesh using proxies
    startTime = timeutils::getTimeMicros();
    referenceMesh.appendQuads(quadSet, gBufferSize, false /* not a reference frame */);
    referenceMesh.createMeshFromProxies(quadSet, gBufferSize, remoteCameraPrev);
    stats.appendQuadsTimeMs = referenceMesh.stats.appendQuadsTimeMs;
    stats.createVertIndTimeMs = referenceMesh.stats.createMeshTimeMs;
    stats.createMeshTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    /*
    ============================
    Create proxies and meshes for the revealed portion of the residual frame capturing disocclusions
    ============================
    */
    startTime = timeutils::getTimeMicros();
    quadsGenerator->createProxiesFromRT(residualFrameRT, currRemoteCamera);
    stats.createQuadsTimeMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    // Transfer revealed proxies to CPU for compression
    auto sizesRevealed = quadSet.writeToMemory(uncompressedQuadsRevealed, uncompressedOffsetsRevealed, params.applyDeltaEncoding);
    residualFrame.numQuadsRevealed = sizesRevealed.numQuads;
    residualFrame.numDepthOffsetsRevealed = sizesRevealed.numDepthOffsets;
    stats.transferTimeMs += quadSet.stats.transferTimeMs;

    // Compress revealed proxies (asynchronous)
    auto quadsRevealedFuture = threadPool->submit_task([&]() {
        return residualFrame.compressAndStoreRevealedQuads(uncompressedQuadsRevealed);
    });
    auto offsetsRevealedFuture = threadPool->submit_task([&]() {
        return residualFrame.compressAndStoreRevealedDepthOffsets(uncompressedOffsetsRevealed);
    });

    // Using GPU buffers, reconstruct revealed mesh using proxies
    startTime = timeutils::getTimeMicros();
    residualMesh.appendQuads(quadSet, gBufferSize);
    residualMesh.createMeshFromProxies(quadSet, gBufferSize, currRemoteCamera);
    stats.appendQuadsTimeMs += residualMesh.stats.appendQuadsTimeMs;
    stats.createVertIndTimeMs += residualMesh.stats.createMeshTimeMs;
    stats.createMeshTimeMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    /*
    ============================
    Wait for asynchronous compression to finish and set resulting data sizes
    ============================
    */
    residualFrame.quadsUpdated.resize(quadsUpdatedFuture.get());
    residualFrame.quadsRevealed.resize(quadsRevealedFuture.get());
    residualFrame.depthOffsetsUpdated.resize(offsetsUpdatedFuture.get());
    residualFrame.depthOffsetsRevealed.resize(offsetsRevealedFuture.get());
    stats.compressTimeMs = residualFrame.getCompressTime();
}
