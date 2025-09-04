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
    stats.timeToGenerateQuadsMs = quadsGenerator->stats.timeToGenerateQuadsMs;
    stats.timeToSimplifyQuadsMs = quadsGenerator->stats.timeToSimplifyQuadsMs;
    stats.timeToGatherQuadsMs = quadsGenerator->stats.timeToGatherQuadsMs;
    stats.timeToCreateQuadsMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    // Transfer updated proxies to CPU for compression
    auto sizes = quadSet.writeToMemory(uncompressedQuads, uncompressedOffsets);
    referenceFrame.numQuads = sizes.numQuads;
    referenceFrame.numDepthOffsets = sizes.numDepthOffsets;
    stats.timeToTransferMs = quadSet.stats.timeToTransferMs;

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
    stats.timeToAppendQuadsMs = referenceMesh.stats.timeToAppendQuadsMs;
    stats.timeToFillQuadIndicesMs = referenceMesh.stats.timeToGatherQuadsMs;
    stats.timeToCreateVertIndMs = referenceMesh.stats.timeToCreateMeshMs;
    stats.timeToCreateMeshMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    /*
    ============================
    Wait for asynchronous compression to finish and set resulting data sizes
    ============================
    */
    referenceFrame.quads.resize(quadsFuture.get());
    referenceFrame.depthOffsets.resize(offsetsFuture.get());
    stats.timeToCompressMs = referenceFrame.getTimeToCompress();
}

void FrameGenerator::updateResidualRenderTargets(
    FrameRenderTarget& residualFrameMaskRT, FrameRenderTarget& residualFrameRT,
    DeferredRenderer& remoteRenderer, Scene& remoteScene,
    Scene& currMeshScene, Scene& prevMeshScene,
    const PerspectiveCamera& currRemoteCamera, const PerspectiveCamera& remoteCameraPrev)
{
    stats.timeToUpdateRTsMs = 0.0;

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

    stats.timeToUpdateRTsMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
}

void FrameGenerator::createResidualFrame(
    const FrameRenderTarget& residualFrameMaskRT, const FrameRenderTarget& residualFrameRT,
    const PerspectiveCamera& currRemoteCamera, const PerspectiveCamera& remoteCameraPrev,
    QuadMesh& referenceMesh, QuadMesh& residualMesh,
    ResidualFrame& residualFrame)
{
    double timeToUpdateRTsMs = stats.timeToUpdateRTsMs;
    stats = { 0 };
    stats.timeToUpdateRTsMs = timeToUpdateRTsMs;

    const glm::vec2 gBufferSize = glm::vec2(residualFrameRT.width, residualFrameRT.height);

    /*
    ============================
    Create proxies and meshes for the masked portion of the residual frame capturing updated geometry
    ============================
    */
    double startTime = timeutils::getTimeMicros();
    quadsGenerator->createProxiesFromRT(residualFrameMaskRT, remoteCameraPrev);
    stats.timeToCreateQuadsMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    // Transfer updated proxies to CPU for compression
    auto sizesUpdated = quadSet.writeToMemory(uncompressedQuads, uncompressedOffsets);
    residualFrame.numQuadsUpdated = sizesUpdated.numQuads;
    residualFrame.numDepthOffsetsUpdated = sizesUpdated.numDepthOffsets;
    stats.timeToTransferMs = quadSet.stats.timeToTransferMs;

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
    stats.timeToAppendQuadsMs = referenceMesh.stats.timeToAppendQuadsMs;
    stats.timeToFillQuadIndicesMs = referenceMesh.stats.timeToGatherQuadsMs;
    stats.timeToCreateVertIndMs = referenceMesh.stats.timeToCreateMeshMs;
    stats.timeToCreateMeshMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    /*
    ============================
    Create proxies and meshes for the revealed portion of the residual frame capturing disocclusions
    ============================
    */
    startTime = timeutils::getTimeMicros();
    quadsGenerator->createProxiesFromRT(residualFrameRT, currRemoteCamera);
    stats.timeToCreateQuadsMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    // Transfer revealed proxies to CPU for compression
    auto sizesRevealed = quadSet.writeToMemory(uncompressedQuadsRevealed, uncompressedOffsetsRevealed);
    residualFrame.numQuadsRevealed = sizesRevealed.numQuads;
    residualFrame.numDepthOffsetsRevealed = sizesRevealed.numDepthOffsets;
    stats.timeToTransferMs += quadSet.stats.timeToTransferMs;

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
    stats.timeToAppendQuadsMs += residualMesh.stats.timeToAppendQuadsMs;
    stats.timeToFillQuadIndicesMs += residualMesh.stats.timeToGatherQuadsMs;
    stats.timeToCreateVertIndMs += residualMesh.stats.timeToCreateMeshMs;
    stats.timeToCreateMeshMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    /*
    ============================
    Wait for asynchronous compression to finish and set resulting data sizes
    ============================
    */
    residualFrame.quadsUpdated.resize(quadsUpdatedFuture.get());
    residualFrame.quadsRevealed.resize(quadsRevealedFuture.get());
    residualFrame.depthOffsetsUpdated.resize(offsetsUpdatedFuture.get());
    residualFrame.depthOffsetsRevealed.resize(offsetsRevealedFuture.get());
    stats.timeToCompressMs = residualFrame.getTimeToCompress();
}
