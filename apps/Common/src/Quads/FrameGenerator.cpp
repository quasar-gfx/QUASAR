#include <algorithm>
#include <Quads/FrameGenerator.h>

using namespace quasar;

FrameGenerator::FrameGenerator(QuadSet& quadSet, DeferredRenderer& remoteRenderer, Scene& remoteScene)
    : quadSet(quadSet)
    , remoteRenderer(remoteRenderer)
    , remoteScene(remoteScene)
    , quadsGenerator(quadSet)
    , maskRT({
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
{}

void FrameGenerator::generateRefFrame(
    const FrameRenderTarget& frameRT, const PerspectiveCamera& remoteCamera,
    QuadMesh& mesh, ReferenceFrame& resultFrame)
{
    const glm::vec2 gBufferSize = glm::vec2(frameRT.width, frameRT.height);

    double startTime = timeutils::getTimeMicros();

    // Create proxies from the current FrameRenderTarget
    quadsGenerator.createProxiesFromRT(frameRT, remoteCamera);
    stats.timeToGenerateQuadsMs = quadsGenerator.stats.timeToGenerateQuadsMs;
    stats.timeToSimplifyQuadsMs = quadsGenerator.stats.timeToSimplifyQuadsMs;
    stats.timeToGatherQuadsMs = quadsGenerator.stats.timeToGatherQuadsMs;
    stats.timeToCreateQuadsMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    // Transfer updated proxies to CPU for compression
    auto sizes = quadSet.mapToCPU(uncompressedQuads, uncompressedOffsets);
    resultFrame.numQuads = sizes.numQuads;
    resultFrame.numDepthOffsets = sizes.numDepthOffsets;
    stats.timeToTransferMs = quadSet.stats.timeToTransferMs;

    // Compress proxies (nonblocking)
    auto quadsFuture = refQuadsCodec.compressAsync(
        uncompressedQuads.data(),
        resultFrame.quads,
        uncompressedQuads.size());
    auto offsetsFuture = refDepthOffsetsCodec.compressAsync(
        uncompressedOffsets.data(),
        resultFrame.depthOffsets,
        uncompressedOffsets.size());

    // Using GPU buffers, create mesh from proxies
    startTime = timeutils::getTimeMicros();
    mesh.appendQuads(quadSet, gBufferSize);
    mesh.createMeshFromProxies(quadSet, gBufferSize, remoteCamera);
    stats.timeToAppendQuadsMs = mesh.stats.timeToAppendQuadsMs;
    stats.timeToFillQuadIndicesMs = mesh.stats.timeToGatherQuadsMs;
    stats.timeToCreateVertIndMs = mesh.stats.timeToCreateMeshMs;
    stats.timeToCreateMeshMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    // Wait for compression to finish and set resulting data sizes
    resultFrame.quads.resize(quadsFuture.get());
    resultFrame.depthOffsets.resize(offsetsFuture.get());
    stats.timeToCompressMs = std::max(refQuadsCodec.stats.timeToCompressMs, refDepthOffsetsCodec.stats.timeToCompressMs);
}

void FrameGenerator::generateResFrame(
    Scene& currScene, Scene& prevScene,
    FrameRenderTarget& resFrameRT,
    const PerspectiveCamera& currRemoteCamera, const PerspectiveCamera& prevRemoteCamera,
    QuadMesh& currMesh, QuadMesh& maskMesh, ResidualFrame& resultFrame)
{
    const glm::vec2 gBufferSize = glm::vec2(resFrameRT.width, resFrameRT.height);
    if (gBufferSize.x != maskRT.width || gBufferSize.y != maskRT.height) {
        maskRT.resize(gBufferSize.x, gBufferSize.y);
    }

    // Generate frame from old camera pose using previous frame as a mask to capture scene changes
    double startTime = timeutils::getTimeMicros();
    {
        // Fill depth buffer with previous generated mesh
        remoteRenderer.pipeline.writeMaskState.disableColorWrites();
        remoteRenderer.drawObjectsNoLighting(prevScene, prevRemoteCamera);

        // Use current generated mesh as a stencil mask
        remoteRenderer.pipeline.stencilState.enableRenderingIntoStencilBuffer(GL_KEEP, GL_KEEP, GL_REPLACE);
        remoteRenderer.pipeline.depthState.depthFunc = GL_EQUAL;
        remoteRenderer.drawObjectsNoLighting(currScene, prevRemoteCamera, GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

        // Render scene using stencil mask; this lets only content that is different pass
        remoteRenderer.pipeline.stencilState.enableRenderingUsingStencilBufferAsMask(GL_NOTEQUAL, 1);
        remoteRenderer.pipeline.depthState.depthFunc = GL_LESS;
        remoteRenderer.pipeline.writeMaskState.enableColorWrites();
        remoteRenderer.drawObjects(remoteScene, prevRemoteCamera, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        remoteRenderer.pipeline.stencilState.restoreStencilState();
        remoteRenderer.copyToFrameRT(maskRT); // Save result into a temporary render target
    }
    double timeToRenderUpdated = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    // Create proxies and meshes for the updated portion of the residual frame
    startTime = timeutils::getTimeMicros();
    quadsGenerator.createProxiesFromRT(maskRT, prevRemoteCamera);
    stats.timeToCreateQuadsMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    // Transfer updated proxies to CPU for compression
    auto sizesUpdated = quadSet.mapToCPU(uncompressedQuadsUpdated, uncompressedOffsetsUpdated);
    resultFrame.numQuadsUpdated = sizesUpdated.numQuads;
    resultFrame.numDepthOffsetsUpdated = sizesUpdated.numDepthOffsets;
    stats.timeToTransferMs = quadSet.stats.timeToTransferMs;

    // Compress proxies (nonblocking)
    auto quadsUpdatedFuture = resQuadsUpdatedCodec.compressAsync(
        uncompressedQuadsUpdated.data(),
        resultFrame.quadsUpdated,
        uncompressedQuadsUpdated.size());
    auto offsetsUpdatedFuture = resDepthOffsetsUpdatedCodec.compressAsync(
        uncompressedOffsetsUpdated.data(),
        resultFrame.depthOffsetsUpdated,
        uncompressedOffsetsUpdated.size());

    // Using GPU buffers, create mesh using proxies
    startTime = timeutils::getTimeMicros();
    currMesh.appendQuads(quadSet, gBufferSize, false);
    currMesh.createMeshFromProxies(quadSet, gBufferSize, prevRemoteCamera);
    stats.timeToAppendQuadsMs = currMesh.stats.timeToAppendQuadsMs;
    stats.timeToFillQuadIndicesMs = currMesh.stats.timeToGatherQuadsMs;
    stats.timeToCreateVertIndMs = currMesh.stats.timeToCreateMeshMs;
    stats.timeToCreateMeshMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    // Generate frame from new camera pose using current frame as a mask to capture disocclusions due to camera movement
    startTime = timeutils::getTimeMicros();
    {
        // Use current generated mesh as a stencil mask
        remoteRenderer.pipeline.stencilState.enableRenderingIntoStencilBuffer(GL_KEEP, GL_KEEP, GL_REPLACE);
        remoteRenderer.pipeline.writeMaskState.disableColorWrites();
        remoteRenderer.drawObjectsNoLighting(currScene, currRemoteCamera);

        // Render scene using stencil mask; this lets only content that is different pass
        remoteRenderer.pipeline.stencilState.enableRenderingUsingStencilBufferAsMask(GL_NOTEQUAL, 1);
        remoteRenderer.pipeline.writeMaskState.enableColorWrites();
        remoteRenderer.drawObjects(remoteScene, currRemoteCamera, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        remoteRenderer.pipeline.stencilState.restoreStencilState();
        remoteRenderer.copyToFrameRT(resFrameRT);
    }
    stats.timeToRenderMaskMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime) + timeToRenderUpdated;

    // Create proxies and meshes for the revealed portion of the residual frame
    startTime = timeutils::getTimeMicros();
    quadsGenerator.createProxiesFromRT(resFrameRT, currRemoteCamera);
    stats.timeToCreateQuadsMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    // Transfer revealed proxies to CPU for compression
    auto sizesRevealed = quadSet.mapToCPU(uncompressedQuadsRevealed, uncompressedOffsetsRevealed);
    resultFrame.numQuadsRevealed = sizesRevealed.numQuads;
    resultFrame.numDepthOffsetsRevealed = sizesRevealed.numDepthOffsets;
    stats.timeToTransferMs += quadSet.stats.timeToTransferMs;

    // Compress proxies (nonblocking)
    auto quadsRevealedFuture = resQuadsRevealedCodec.compressAsync(
        uncompressedQuadsRevealed.data(),
        resultFrame.quadsRevealed,
        uncompressedQuadsRevealed.size());
    auto offsetsRevealedFuture = resDepthOffsetsRevealedCodec.compressAsync(
        uncompressedOffsetsRevealed.data(),
        resultFrame.depthOffsetsRevealed,
        uncompressedOffsetsRevealed.size());

    // Using GPU buffers, create mesh using proxies
    startTime = timeutils::getTimeMicros();
    maskMesh.appendQuads(quadSet, gBufferSize);
    maskMesh.createMeshFromProxies(quadSet, gBufferSize, currRemoteCamera);
    stats.timeToAppendQuadsMs += maskMesh.stats.timeToAppendQuadsMs;
    stats.timeToFillQuadIndicesMs += maskMesh.stats.timeToGatherQuadsMs;
    stats.timeToCreateVertIndMs += maskMesh.stats.timeToCreateMeshMs;
    stats.timeToCreateMeshMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    // Wait for compression to finish and set resulting data sizes
    resultFrame.quadsUpdated.resize(quadsUpdatedFuture.get());
    resultFrame.depthOffsetsUpdated.resize(offsetsUpdatedFuture.get());
    resultFrame.quadsRevealed.resize(quadsRevealedFuture.get());
    resultFrame.depthOffsetsRevealed.resize(offsetsRevealedFuture.get());
    stats.timeToCompressMs = std::max(
        std::max(resQuadsUpdatedCodec.stats.timeToCompressMs, resDepthOffsetsUpdatedCodec.stats.timeToCompressMs),
        std::max(resQuadsRevealedCodec.stats.timeToCompressMs, resDepthOffsetsRevealedCodec.stats.timeToCompressMs)
    );
}
