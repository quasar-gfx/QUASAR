#include <Quads/FrameGenerator.h>

unsigned int FrameGenerator::generateIFrame(
        const GBuffer &gBuffer, const GBuffer &gBufferHighRes,
        const PerspectiveCamera &remoteCamera,
        QuadsGenerator &quadsGenerator, MeshFromQuads &meshFromQuads, const Mesh &mesh,
        unsigned int &numProxies, unsigned int &numDepthOffsets,
        bool compress
    ) {
    const glm::vec2 gBufferSize = glm::vec2(gBuffer.width, gBuffer.height);

    double startTimeTotal = timeutils::getTimeMicros();
    double startTime = startTimeTotal;

    // create proxies from the current gBuffer
    auto sizes = quadsGenerator.createProxiesFromGBuffer(gBuffer, gBufferHighRes, remoteCamera);
    numProxies = sizes.numProxies;
    numDepthOffsets = sizes.numDepthOffsets;
    stats.timeToCreateProxies = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
    stats.timeToGenerateQuads = quadsGenerator.stats.timeToGenerateQuadsMs;
    stats.timeToSimplifyQuads = quadsGenerator.stats.timeToSimplifyQuadsMs;
    stats.timeToFillOutputQuads = quadsGenerator.stats.timeToFillOutputQuadsMs;

    // create mesh from the proxies
    startTime = timeutils::getTimeMicros();
    meshFromQuads.appendProxies(
        gBufferSize,
        numProxies, quadsGenerator.outputQuadBuffers
    );
    meshFromQuads.createMeshFromProxies(
        gBufferSize,
        numProxies, quadsGenerator.depthOffsets,
        remoteCamera,
        mesh
    );
    stats.timeToAppendProxies = meshFromQuads.stats.timeToAppendProxiesMs;
    stats.timeToFillQuadIndices = meshFromQuads.stats.timeToFillOutputQuadsMs;
    stats.timeToCreateVertInd = meshFromQuads.stats.timeToCreateMeshMs;
    stats.timeToCreateMesh = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    return quadsGenerator.saveQuadsToMemory(compressedQuads, compress) +
           numDepthOffsets * sizeof(uint16_t) / 8;
}

unsigned int FrameGenerator::generatePFrame(
        ForwardRenderer &remoteRenderer, const Scene &remoteScene, const Scene &currScene, const Scene &prevScene,
        GBuffer &gBufferHighRes, GBuffer &gBufferMaskHighRes,
        GBuffer &gBufferLowRes, GBuffer &gBufferMaskLowRes,
        const PerspectiveCamera &currRemoteCamera, const PerspectiveCamera &prevRemoteCamera,
        QuadsGenerator &quadsGenerator, MeshFromQuads &meshFromQuads, MeshFromQuads &meshFromQuadsMask,
        const Mesh &currMesh, const Mesh &maskMesh,
        unsigned int &numProxies, unsigned int &numDepthOffsets,
        bool compress
    ) {
    const glm::vec2 gBufferSize = glm::vec2(gBufferLowRes.width, gBufferLowRes.height);
    unsigned int outputSize = 0;

    double startTimeTotal = timeutils::getTimeMicros();
    double startTime = startTimeTotal;

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
        remoteRenderer.gBuffer.blitToGBuffer(gBufferLowRes);
        remoteRenderer.gBuffer.blitToGBuffer(gBufferHighRes);
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
        remoteRenderer.gBuffer.blitToGBuffer(gBufferMaskLowRes);
        remoteRenderer.gBuffer.blitToGBuffer(gBufferMaskHighRes);

        stats.timeToRenderMasks = timeutils::microsToMillis(timeutils::getTimeMicros() - startTimeTotal);
    }

    // create proxies and meshes
    {
        auto sizes = quadsGenerator.createProxiesFromGBuffer(gBufferLowRes, gBufferHighRes, prevRemoteCamera);
        numProxies = sizes.numProxies;
        numDepthOffsets = sizes.numDepthOffsets;
        stats.timeToCreateProxies = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        meshFromQuads.appendProxies(
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
        stats.timeToAppendProxies = meshFromQuads.stats.timeToAppendProxiesMs;
        stats.timeToCreateVertInd = meshFromQuads.stats.timeToCreateMeshMs;

        outputSize += quadsGenerator.saveQuadsToMemory(compressedQuads, compress);
    }

    // bool oldExpandEdges = quadsGenerator.expandEdges;
    // quadsGenerator.expandEdges = true; // must expand edges to fill small holes in quad boundaries when creating mask

    {
        auto sizes = quadsGenerator.createProxiesFromGBuffer(gBufferMaskLowRes, gBufferMaskHighRes, currRemoteCamera);
        numProxies += sizes.numProxies;
        numDepthOffsets += sizes.numDepthOffsets;

        meshFromQuadsMask.appendProxies(
            gBufferSize,
            sizes.numProxies, quadsGenerator.outputQuadBuffers
        );
        meshFromQuadsMask.createMeshFromProxies(
            gBufferSize,
            sizes.numProxies, quadsGenerator.depthOffsets,
            currRemoteCamera,
            maskMesh
        );
        stats.timeToAppendProxies += meshFromQuadsMask.stats.timeToAppendProxiesMs;
        stats.timeToCreateVertInd += meshFromQuadsMask.stats.timeToCreateMeshMs;

        outputSize += quadsGenerator.saveQuadsToMemory(compressedQuads, compress);
        outputSize += numDepthOffsets * sizeof(uint16_t) / 8;
    }

    // restore expand edges
    // quadsGenerator.expandEdges = oldExpandEdges;

    return outputSize;
}
