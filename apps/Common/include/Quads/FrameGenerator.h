#ifndef FRAME_GENERATOR_H
#define FRAME_GENERATOR_H

#include <Renderers/ForwardRenderer.h>
#include <RenderTargets/GBuffer.h>

#include <Quads/QuadsBuffers.h>
#include <Quads/DepthOffsets.h>
#include <Quads/QuadsGenerator.h>
#include <Quads/MeshFromQuads.h>

class FrameGenerator {
public:
    FrameGenerator() = default;

    struct Stats {
        double timeToCreateProxies = 0.0f;
        double timeToCreateMesh = 0.0f;
        double timeToGenerateQuads = 0.0f;
        double timeToSimplifyQuads = 0.0f;
        double timeToFillOutputQuads = 0.0f;
        double timeToAppendProxies = 0.0f;
        double timeToFillQuadIndices = 0.0f;
        double timeToCreateVertInd = 0.0f;
        double timeToRenderMasks = 0.0f;
    } stats;

    void generateIFrame(
            const GBuffer &gBuffer, const PerspectiveCamera &remoteCamera,
            QuadsGenerator &quadsGenerator, MeshFromQuads &meshFromQuads, const Mesh &mesh,
            unsigned int &numProxies, unsigned int &numDepthOffsets) {
        const glm::vec2 gBufferSize = glm::vec2(gBuffer.width, gBuffer.height);

        double startTimeTotal = timeutils::getTimeMicros();
        double startTime = startTimeTotal;

        // create proxies from the current gBuffer
        auto sizes = quadsGenerator.createProxiesFromGBuffer(gBuffer, remoteCamera);
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
        stats.timeToFillQuadIndices = meshFromQuads.stats.timeToFillOutputQuadsMs;
        stats.timeToCreateVertInd = meshFromQuads.stats.timeToCreateMeshMs;
        stats.timeToCreateMesh = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
    }

    void generatePFrame(
            ForwardRenderer &remoteRenderer, const Scene &remoteScene, const Scene &currScene, const Scene &prevScene,
            GBuffer &gBuffer, GBuffer &gBufferMask,
            const PerspectiveCamera &currRemoteCamera, const PerspectiveCamera &prevRemoteCamera,
            QuadsGenerator &quadsGenerator, MeshFromQuads &meshFromQuads, MeshFromQuads &meshFromQuadsMask,
            const Mesh &currMesh, const Mesh &maskMesh,
            unsigned int &numProxies, unsigned int &numDepthOffsets) {
        // at this point, the current mesh is filled with the current frame

        const glm::vec2 gBufferSize = glm::vec2(gBuffer.width, gBuffer.height);

        double startTimeTotal = timeutils::getTimeMicros();
        double startTime = startTimeTotal;

        /*
        ============================
        Generate frame using previous frame as a mask for animations
        ============================
        */
        {
            // first, draw the previous mesh at the previous camera view, filling depth buffer
            remoteRenderer.pipeline.writeMaskState.disableColorWrites();
            remoteRenderer.drawObjectsNoLighting(prevScene, prevRemoteCamera);

            // then, render the current mesh scene into stencil buffer, using the depth buffer from the prev mesh scene
            // this should draw fragments in the current mesh that are not occluded by the prev mesh scene, setting
            // the stencil buffer to 1 where the depth of the curr mesh is the same as the prev mesh scene
            remoteRenderer.pipeline.stencilState.enableRenderingIntoStencilBuffer(GL_KEEP, GL_KEEP, GL_REPLACE);
            remoteRenderer.pipeline.depthState.depthFunc = GL_EQUAL;
            remoteRenderer.drawObjectsNoLighting(currScene, prevRemoteCamera, GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

            // now, render the full remote scene using the stencil buffer as a mask
            // with this, at values where stencil buffer is 1, remoteScene should render
            remoteRenderer.pipeline.stencilState.enableRenderingUsingStencilBufferAsMask(GL_NOTEQUAL, 1);
            remoteRenderer.pipeline.depthState.depthFunc = GL_LESS;
            remoteRenderer.pipeline.writeMaskState.enableColorWrites();
            remoteRenderer.drawObjects(remoteScene, prevRemoteCamera, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            remoteRenderer.pipeline.stencilState.restoreStencilState();

            remoteRenderer.gBuffer.blitToGBuffer(gBuffer);
        }

        /*
        ============================
        Generate frame using current frame as a mask for movement
        ============================
        */
        {
            // draw current meshes at current remoteCamera view, filling stencil buffer with 1
            remoteRenderer.pipeline.stencilState.enableRenderingIntoStencilBuffer(GL_KEEP, GL_KEEP, GL_REPLACE);
            remoteRenderer.pipeline.writeMaskState.disableColorWrites();
            remoteRenderer.drawObjectsNoLighting(currScene, currRemoteCamera);

            // render remoteScene using stencil buffer as a mask
            // at values where stencil buffer is not 1, remoteScene should render
            remoteRenderer.pipeline.stencilState.enableRenderingUsingStencilBufferAsMask(GL_NOTEQUAL, 1);
            remoteRenderer.pipeline.writeMaskState.enableColorWrites();
            remoteRenderer.drawObjects(remoteScene, currRemoteCamera, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            remoteRenderer.pipeline.stencilState.restoreStencilState();

            remoteRenderer.gBuffer.blitToGBuffer(gBufferMask);
        }

        stats.timeToRenderMasks = timeutils::microsToMillis(timeutils::getTimeMicros() - startTimeTotal);

        {
            // create proxies
            auto sizes = quadsGenerator.createProxiesFromGBuffer(gBuffer, prevRemoteCamera);
            numProxies = sizes.numProxies;
            stats.timeToCreateProxies = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
            stats.timeToGenerateQuads = quadsGenerator.stats.timeToGenerateQuadsMs;
            stats.timeToSimplifyQuads = quadsGenerator.stats.timeToSimplifyQuadsMs;
            stats.timeToFillOutputQuads = quadsGenerator.stats.timeToFillOutputQuadsMs;

            // create mesh from proxies
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
            stats.timeToFillQuadIndices = meshFromQuads.stats.timeToFillOutputQuadsMs;
            stats.timeToCreateVertInd = meshFromQuads.stats.timeToCreateMeshMs;
        }
        {
            // create proxies
            startTime = timeutils::getTimeMicros();
            auto sizes = quadsGenerator.createProxiesFromGBuffer(gBufferMask, currRemoteCamera);
            numProxies += sizes.numProxies;
            stats.timeToCreateProxies += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
            stats.timeToGenerateQuads += quadsGenerator.stats.timeToGenerateQuadsMs;
            stats.timeToSimplifyQuads += quadsGenerator.stats.timeToSimplifyQuadsMs;
            stats.timeToFillOutputQuads += quadsGenerator.stats.timeToFillOutputQuadsMs;

            // create mesh from proxies
            meshFromQuadsMask.appendProxies(
                gBufferSize,
                sizes.numProxies, quadsGenerator.outputQuadBuffers,
                true
            );
            meshFromQuadsMask.createMeshFromProxies(
                gBufferSize,
                sizes.numProxies, quadsGenerator.depthOffsets,
                currRemoteCamera,
                maskMesh
            );
            stats.timeToAppendProxies += meshFromQuadsMask.stats.timeToAppendProxiesMs;
            stats.timeToCreateVertInd += meshFromQuadsMask.stats.timeToCreateMeshMs;
        }
    }
};

#endif // FRAME_GENERATOR_H
