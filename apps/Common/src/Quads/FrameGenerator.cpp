#include <Quads/FrameGenerator.h>

using namespace quasar;

unsigned int FrameGenerator::generateRefFrame(
        const FrameRenderTarget &frameRT,
        const PerspectiveCamera &remoteCamera,
        const Mesh &mesh,
        std::vector<char> &quads, std::vector<char> &depthOffsets,
        unsigned int &numProxies, unsigned int &numDepthOffsets,
        bool compress
    ) {
    const glm::vec2 gBufferSize = glm::vec2(frameRT.width, frameRT.height);
    unsigned int outputSize = 0;

    double startTime = timeutils::getTimeMicros();

    // create proxies from the current FrameRenderTarget
    quadsGenerator.createProxiesFromGBuffer(frameRT, remoteCamera);
    stats.timeToCreateProxiesMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
    stats.timeToGenerateQuadsMs = quadsGenerator.stats.timeToGenerateQuadsMs;
    stats.timeToSimplifyQuadsMs = quadsGenerator.stats.timeToSimplifyQuadsMs;
    stats.timeToGatherQuadsMs = quadsGenerator.stats.timeToGatherQuadsMs;

    outputSize += quadsGenerator.saveQuadsToMemory(quads, compress);
    outputSize += quadsGenerator.saveDepthOffsetsToMemory(depthOffsets, compress) / 8;
    stats.timeToCompress = quadsGenerator.outputQuadBuffers.stats.timeToCompressMs +
                           quadsGenerator.depthOffsets.stats.timeToCompressMs;

    // create mesh from the proxies
    startTime = timeutils::getTimeMicros();
    auto sizes = quadsGenerator.getBufferSizes();
    numProxies = sizes.numProxies;
    numDepthOffsets = sizes.numDepthOffsets;
    meshFromQuads.appendQuads(
        gBufferSize,
        numProxies, quadsGenerator.outputQuadBuffers
    );
    meshFromQuads.createMeshFromProxies(
        gBufferSize,
        numProxies, quadsGenerator.depthOffsets,
        remoteCamera,
        mesh
    );
    stats.timeToAppendQuadsMs = meshFromQuads.stats.timeToAppendQuadsMs;
    stats.timeToFillQuadIndicesMs = meshFromQuads.stats.timeToGatherQuadsMs;
    stats.timeToCreateVertIndMs = meshFromQuads.stats.timeToCreateMeshMs;
    stats.timeToCreateMeshMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    return outputSize;
}

unsigned int FrameGenerator::generateResFrame(
        const Scene &currScene, const Scene &prevScene,
        FrameRenderTarget &frameRT, FrameRenderTarget &maskFrameRT,
        const PerspectiveCamera &currRemoteCamera, const PerspectiveCamera &prevRemoteCamera,
        const Mesh &currMesh, const Mesh &maskMesh,
        std::vector<char> &quads, std::vector<char> &depthOffsets,
        unsigned int &numProxies, unsigned int &numDepthOffsets,
        bool compress
    ) {
    const glm::vec2 gBufferSize = glm::vec2(frameRT.width, frameRT.height);
    unsigned int outputSize = 0;

    double startTime = timeutils::getTimeMicros();

    // generate frame using previous frame as a mask for animations
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

    // generate frame using current frame as a mask for movement
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

    // create proxies and meshes for the updated keyframe
    {
        startTime = timeutils::getTimeMicros();
        quadsGenerator.createProxiesFromGBuffer(frameRT, prevRemoteCamera);
        stats.timeToCreateProxiesMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        outputSize += quadsGenerator.saveQuadsToMemory(quads, compress);
        outputSize += quadsGenerator.saveDepthOffsetsToMemory(depthOffsets, compress) / 8;
        stats.timeToCompress = quadsGenerator.outputQuadBuffers.stats.timeToCompressMs +
                               quadsGenerator.depthOffsets.stats.timeToCompressMs;

        startTime = timeutils::getTimeMicros();
        auto sizes = quadsGenerator.getBufferSizes();
        numProxies = sizes.numProxies;
        numDepthOffsets = sizes.numDepthOffsets;
        meshFromQuads.appendQuads(
            gBufferSize,
            sizes.numProxies, quadsGenerator.outputQuadBuffers,
            false
        );
        meshFromQuads.createMeshFromProxies(
            gBufferSize,
            sizes.numProxies, quadsGenerator.depthOffsets,
            prevRemoteCamera,
            currMesh
        );
        stats.timeToAppendQuadsMs = meshFromQuads.stats.timeToAppendQuadsMs;
        stats.timeToFillQuadIndicesMs = meshFromQuads.stats.timeToGatherQuadsMs;
        stats.timeToCreateVertIndMs = meshFromQuads.stats.timeToCreateMeshMs;
        stats.timeToCreateMeshMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
    }

    // create proxies and meshes for the updated mask
    {
        startTime = timeutils::getTimeMicros();
        quadsGenerator.createProxiesFromGBuffer(maskFrameRT, currRemoteCamera);
        stats.timeToCreateProxiesMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        outputSize += quadsGenerator.saveQuadsToMemory(quads, compress);
        outputSize += quadsGenerator.saveDepthOffsetsToMemory(depthOffsets, compress) / 8;
        stats.timeToCompress += quadsGenerator.outputQuadBuffers.stats.timeToCompressMs +
                                quadsGenerator.depthOffsets.stats.timeToCompressMs;

        startTime = timeutils::getTimeMicros();
        auto sizes = quadsGenerator.getBufferSizes();
        numProxies += sizes.numProxies;
        numDepthOffsets += sizes.numDepthOffsets;
        meshFromQuadsMask.appendQuads(
            gBufferSize,
            sizes.numProxies, quadsGenerator.outputQuadBuffers
        );
        meshFromQuadsMask.createMeshFromProxies(
            gBufferSize,
            sizes.numProxies, quadsGenerator.depthOffsets,
            currRemoteCamera,
            maskMesh
        );
        stats.timeToAppendQuadsMs += meshFromQuads.stats.timeToAppendQuadsMs;
        stats.timeToFillQuadIndicesMs += meshFromQuads.stats.timeToGatherQuadsMs;
        stats.timeToCreateVertIndMs += meshFromQuads.stats.timeToCreateMeshMs;
        stats.timeToCreateMeshMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
    }

    return outputSize;
}
