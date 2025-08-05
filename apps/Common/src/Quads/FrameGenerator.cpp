#include <Quads/FrameGenerator.h>

using namespace quasar;

FrameGenerator::FrameGenerator(QuadFrame& quadFrame, DeferredRenderer& remoteRenderer, Scene& remoteScene)
    : quadFrame(quadFrame)
    , remoteRenderer(remoteRenderer)
    , remoteScene(remoteScene)
    , quadsGenerator(quadFrame)
    , meshFromQuads(quadFrame)
    , meshFromQuadsMask(quadFrame, meshFromQuads.maxProxies / 4) // Mask mesh can have fewer proxies
{}

uint FrameGenerator::generateRefFrame(
    const FrameRenderTarget& frameRT,
    const PerspectiveCamera& remoteCamera,
    const Mesh& mesh,
    uint& numProxies, uint& numDepthOffsets)
{
    const glm::vec2 gBufferSize = glm::vec2(frameRT.width, frameRT.height);
    uint outputSize = 0;

    double startTime = timeutils::getTimeMicros();

    // Create proxies from the current FrameRenderTarget
    quadsGenerator.createProxiesFromRT(frameRT, remoteCamera);
    stats.timeToCreateProxiesMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
    stats.timeToGenerateQuadsMs = quadsGenerator.stats.timeToGenerateQuadsMs;
    stats.timeToSimplifyQuadsMs = quadsGenerator.stats.timeToSimplifyQuadsMs;
    stats.timeToGatherQuadsMs = quadsGenerator.stats.timeToGatherQuadsMs;

    auto [quadsSize, depthOffsetsSize] = quadFrame.copyToMemory();
    outputSize += quadsSize + depthOffsetsSize;
    stats.timeToCompress = quadFrame.stats.timeToCompressMs;

    numProxies = quadFrame.getNumQuads();
    numDepthOffsets = quadFrame.getNumDepthOffsets();

    // Create mesh from the proxies
    startTime = timeutils::getTimeMicros();
    {
        meshFromQuads.appendQuads(quadFrame, gBufferSize);
        meshFromQuads.createMeshFromProxies(quadFrame, gBufferSize, remoteCamera, mesh);
    }
    stats.timeToAppendQuadsMs = meshFromQuads.stats.timeToAppendQuadsMs;
    stats.timeToFillQuadIndicesMs = meshFromQuads.stats.timeToGatherQuadsMs;
    stats.timeToCreateVertIndMs = meshFromQuads.stats.timeToCreateMeshMs;
    stats.timeToCreateMeshMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    return outputSize;
}

uint FrameGenerator::generateResFrame(
    Scene& currScene, Scene& prevScene,
    FrameRenderTarget& frameRT, FrameRenderTarget& maskFrameRT,
    const PerspectiveCamera& currRemoteCamera, const PerspectiveCamera& prevRemoteCamera,
    const Mesh& currMesh, const Mesh& maskMesh,
    uint& numProxies, uint& numDepthOffsets)
{
    const glm::vec2 gBufferSize = glm::vec2(frameRT.width, frameRT.height);
    uint outputSize = 0;

    double startTime = timeutils::getTimeMicros();

    // Generate frame using previous frame as a mask for animations
    {
        remoteRenderer.pipeline.writeMaskState.disableColorWrites();
        remoteRenderer.drawObjectsNoLighting(prevScene, prevRemoteCamera);

        remoteRenderer.pipeline.stencilState.enableRenderingIntoStencilBuffer(GL_KEEP, GL_KEEP, GL_REPLACE);
        remoteRenderer.pipeline.depthState.depthFunc = GL_EQUAL;
        remoteRenderer.drawObjectsNoLighting(currScene, prevRemoteCamera, GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

        remoteRenderer.pipeline.stencilState.enableRenderingUsingStencilBufferAsMask(GL_NOTEQUAL, 1);
        remoteRenderer.pipeline.depthState.depthFunc = GL_LESS;
        remoteRenderer.pipeline.writeMaskState.enableColorWrites();
        remoteRenderer.drawObjects(remoteScene, prevRemoteCamera, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        remoteRenderer.pipeline.stencilState.restoreStencilState();
        remoteRenderer.copyToFrameRT(frameRT);
    }

    // Generate frame using current frame as a mask for movement
    {
        remoteRenderer.pipeline.stencilState.enableRenderingIntoStencilBuffer(GL_KEEP, GL_KEEP, GL_REPLACE);
        remoteRenderer.pipeline.writeMaskState.disableColorWrites();
        remoteRenderer.drawObjectsNoLighting(currScene, currRemoteCamera);

        remoteRenderer.pipeline.stencilState.enableRenderingUsingStencilBufferAsMask(GL_NOTEQUAL, 1);
        remoteRenderer.pipeline.writeMaskState.enableColorWrites();
        remoteRenderer.drawObjects(remoteScene, currRemoteCamera, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        remoteRenderer.pipeline.stencilState.restoreStencilState();
        remoteRenderer.copyToFrameRT(maskFrameRT);
    }

    stats.timeToRenderMaskMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    // Create proxies and meshes for the updated keyframe
    {
        startTime = timeutils::getTimeMicros();
        quadsGenerator.createProxiesFromRT(frameRT, prevRemoteCamera);
        stats.timeToCreateProxiesMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        numProxies = quadFrame.getNumQuads();
        numDepthOffsets = quadFrame.getNumDepthOffsets();

        auto [quadsSize, depthOffsetsSize] = quadFrame.copyToMemory();
        outputSize += quadsSize + depthOffsetsSize;
        stats.timeToCompress = quadFrame.stats.timeToCompressMs;

        startTime = timeutils::getTimeMicros();
        {
            meshFromQuads.appendQuads(quadFrame, gBufferSize, false);
            meshFromQuads.createMeshFromProxies(quadFrame, gBufferSize, prevRemoteCamera, currMesh);
        }
        stats.timeToAppendQuadsMs = meshFromQuads.stats.timeToAppendQuadsMs;
        stats.timeToFillQuadIndicesMs = meshFromQuads.stats.timeToGatherQuadsMs;
        stats.timeToCreateVertIndMs = meshFromQuads.stats.timeToCreateMeshMs;
        stats.timeToCreateMeshMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
    }

    // Create proxies and meshes for the updated mask
    {
        startTime = timeutils::getTimeMicros();
        quadsGenerator.createProxiesFromRT(maskFrameRT, currRemoteCamera);
        stats.timeToCreateProxiesMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        numProxies += quadFrame.getNumQuads();
        numDepthOffsets += quadFrame.getNumDepthOffsets();

        auto [quadsSize, depthOffsetsSize] = quadFrame.copyToMemory();
        outputSize += quadsSize + depthOffsetsSize;
        stats.timeToCompress = quadFrame.stats.timeToCompressMs;

        startTime = timeutils::getTimeMicros();
        {
            meshFromQuadsMask.appendQuads(quadFrame, gBufferSize);
            meshFromQuadsMask.createMeshFromProxies(quadFrame, gBufferSize, currRemoteCamera, maskMesh);
        }
        stats.timeToAppendQuadsMs += meshFromQuadsMask.stats.timeToAppendQuadsMs;
        stats.timeToFillQuadIndicesMs += meshFromQuadsMask.stats.timeToGatherQuadsMs;
        stats.timeToCreateVertIndMs += meshFromQuadsMask.stats.timeToCreateMeshMs;
        stats.timeToCreateMeshMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
    }

    return outputSize;
}
