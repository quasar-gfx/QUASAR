#ifndef QUADS_SIMULATOR_H
#define QUADS_SIMULATOR_H

#include <PostProcessing/ToneMapper.h>
#include <PostProcessing/ShowNormalsEffect.h>

#include <DepthMesh.h>

#include <Quads/QuadMaterial.h>
#include <Quads/FrameGenerator.h>

namespace quasar {

class QuadsSimulator {
public:
    const std::vector<glm::vec4> colors = {
        glm::vec4(1.0f, 1.0f, 0.0f, 1.0f), // primary view color is yellow
        glm::vec4(0.0f, 0.0f, 1.0f, 1.0f),
        glm::vec4(0.0f, 1.0f, 0.0f, 1.0f),
        glm::vec4(1.0f, 0.5f, 0.5f, 1.0f),
        glm::vec4(0.5f, 0.0f, 0.0f, 1.0f),
        glm::vec4(0.0f, 1.0f, 1.0f, 1.0f),
        glm::vec4(1.0f, 0.0f, 0.0f, 1.0f),
        glm::vec4(0.0f, 0.5f, 0.0f, 1.0f),
        glm::vec4(0.0f, 0.0f, 0.5f, 1.0f),
        glm::vec4(0.5f, 0.0f, 0.5f, 1.0f),
    };

    // Reference frame
    FrameRenderTarget refFrameRT;
    std::vector<Mesh> refFrameMeshes;
    std::vector<Node> refFrameNodes;

    // Mask frame (residual frame)
    FrameRenderTarget maskFrameRT;
    FrameRenderTarget maskTempRT;
    Mesh maskFrameMesh;
    Node maskFrameNode;

    // Local objects
    std::vector<Node> refFrameNodesLocal;
    std::vector<Node> refFrameWireframesLocal;
    Node maskFrameWireframeNodesLocal;

    DepthMesh depthMesh;
    Node depthNode;

    FrameRenderTarget copyRT;

    int currMeshIndex = 0, prevMeshIndex = 1;

    uint maxVertices = MAX_NUM_PROXIES * NUM_SUB_QUADS * VERTICES_IN_A_QUAD;
    uint maxIndices = MAX_NUM_PROXIES * NUM_SUB_QUADS * INDICES_IN_A_QUAD;
    uint maxVerticesDepth;

    struct Stats {
        double totalRenderTime = 0.0;
        double totalCreateProxiesTime = 0.0;
        double totalGenQuadMapTime = 0.0;
        double totalSimplifyTime = 0.0;
        double totalGatherQuadsTime = 0.0;
        double totalCreateMeshTime = 0.0;
        double totalAppendQuadsTime = 0.0;
        double totalFillQuadsIndiciesTime = 0.0;
        double totalCreateVertIndTime = 0.0;
        double totalGenDepthTime = 0.0;
        double totalCompressTime = 0.0;

        uint totalProxies = 0;
        uint totalDepthOffsets = 0;
        double compressedSizeBytes = 0;
    } stats;

    QuadsSimulator(QuadFrame& quadFrame, const PerspectiveCamera& remoteCamera, FrameGenerator& frameGenerator)
        : quadFrame(quadFrame)
        , frameGenerator(frameGenerator)
        , maxVerticesDepth(quadFrame.getSize().x * quadFrame.getSize().y)
        , refFrameRT({
            .width = quadFrame.getSize().x,
            .height = quadFrame.getSize().y,
            .internalFormat = GL_RGBA16F,
            .format = GL_RGBA,
            .type = GL_HALF_FLOAT,
            .wrapS = GL_CLAMP_TO_EDGE,
            .wrapT = GL_CLAMP_TO_EDGE,
            .minFilter = GL_NEAREST,
            .magFilter = GL_NEAREST,
        })
        , maskFrameRT({
            .width = quadFrame.getSize().x,
            .height = quadFrame.getSize().y,
            .internalFormat = GL_RGBA16F,
            .format = GL_RGBA,
            .type = GL_HALF_FLOAT,
            .wrapS = GL_CLAMP_TO_EDGE,
            .wrapT = GL_CLAMP_TO_EDGE,
            .minFilter = GL_NEAREST,
            .magFilter = GL_NEAREST,
        })
        , maskTempRT({
            .width = quadFrame.getSize().x,
            .height = quadFrame.getSize().y,
            .internalFormat = GL_RGBA16F,
            .format = GL_RGBA,
            .type = GL_HALF_FLOAT,
            .wrapS = GL_CLAMP_TO_EDGE,
            .wrapT = GL_CLAMP_TO_EDGE,
            .minFilter = GL_NEAREST,
            .magFilter = GL_NEAREST,
        })
        , copyRT({
            .width = quadFrame.getSize().x,
            .height = quadFrame.getSize().y,
            .internalFormat = GL_RGBA16F,
            .format = GL_RGBA,
            .type = GL_HALF_FLOAT,
            .wrapS = GL_CLAMP_TO_EDGE,
            .wrapT = GL_CLAMP_TO_EDGE,
            .minFilter = GL_NEAREST,
            .magFilter = GL_NEAREST,
        })
        , depthMesh(quadFrame.getSize(), glm::vec4(0.0f, 1.0f, 0.0f, 1.0f))
        , meshScenes(2)
    {
        remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());
        remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());

        refFrameMeshes.reserve(2);
        refFrameNodes.reserve(2);
        refFrameNodesLocal.reserve(2);
        refFrameWireframesLocal.reserve(2);

        MeshSizeCreateParams meshParams({
            .maxVertices = maxVertices,
            .maxIndices = maxIndices,
            .vertexSize = sizeof(QuadVertex),
            .attributes = QuadVertex::getVertexInputAttributes(),
            .material = new QuadMaterial({ .baseColorTexture = &refFrameRT.colorBuffer ,}),
            .usage = GL_DYNAMIC_DRAW,
            .indirectDraw = true
        });
        for (int i = 0; i < 2; i++) {
            refFrameMeshes.emplace_back(meshParams);

            refFrameNodes.emplace_back(&refFrameMeshes[i]);
            refFrameNodes[i].frustumCulled = false;
            meshScenes[i].addChildNode(&refFrameNodes[i]);

            refFrameNodesLocal.emplace_back(&refFrameMeshes[i]);
            refFrameNodesLocal[i].frustumCulled = false;

            refFrameWireframesLocal.emplace_back(&refFrameMeshes[i]);
            refFrameWireframesLocal[i].frustumCulled = false;
            refFrameWireframesLocal[i].wireframe = true;
            refFrameWireframesLocal[i].visible = false;
            refFrameWireframesLocal[i].overrideMaterial = &wireframeMaterial;
        }

        // We can use less vertices and indicies for the mask since it will be sparse
        meshParams.maxVertices /= 4;
        meshParams.maxIndices /= 4;
        meshParams.material = new QuadMaterial({ .baseColorTexture = &maskFrameRT.colorBuffer });
        maskFrameMesh = Mesh(meshParams);
        maskFrameNode.setEntity(&maskFrameMesh);
        maskFrameNode.frustumCulled = false;

        maskFrameWireframeNodesLocal.setEntity(&maskFrameMesh);
        maskFrameWireframeNodesLocal.frustumCulled = false;
        maskFrameWireframeNodesLocal.wireframe = true;
        maskFrameWireframeNodesLocal.visible = false;
        maskFrameWireframeNodesLocal.overrideMaterial = &maskWireframeMaterial;

        depthNode.setEntity(&depthMesh);
        depthNode.frustumCulled = false;
        depthNode.visible = false;
        depthNode.primativeType = GL_POINTS;
    }
    ~QuadsSimulator() = default;

    void addMeshesToScene(Scene& localScene) {
        for (int i = 0; i < 2; i++) {
            localScene.addChildNode(&refFrameNodesLocal[i]);
            localScene.addChildNode(&refFrameWireframesLocal[i]);
        }
        localScene.addChildNode(&maskFrameNode);
        localScene.addChildNode(&maskFrameWireframeNodesLocal);
        localScene.addChildNode(&depthNode);
    }

    void generateFrame(
            const PerspectiveCamera& remoteCamera, Scene& remoteScene,
            DeferredRenderer& remoteRenderer,
            bool generateResFrame = false, bool showNormals = false, bool showDepth = false) {
        double startTime = timeutils::getTimeMicros();

        auto& remoteCameraToUse = generateResFrame ? remoteCameraPrev : remoteCamera;

        // Reset stats
        stats = { 0 };

        // Render all objects in remoteScene normally
        remoteRenderer.drawObjects(remoteScene, remoteCameraToUse);
        if (!showNormals) {
            remoteRenderer.copyToFrameRT(refFrameRT);
            toneMapper.drawToRenderTarget(remoteRenderer, copyRT);
        }
        else {
            showNormalsEffect.drawToRenderTarget(remoteRenderer, refFrameRT);
        }
        stats.totalRenderTime += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        /*
        ============================
        Generate Reference Frame
        ============================
        */
        uint numProxies = 0, numDepthOffsets = 0;
        auto& quadsGenerator = frameGenerator.quadsGenerator;
        quadsGenerator.params.expandEdges = false;
        stats.compressedSizeBytes = frameGenerator.generateRefFrame(
            refFrameRT,
            remoteCameraToUse,
            refFrameMeshes[currMeshIndex],
            numProxies, numDepthOffsets
        );
        stats.totalGenQuadMapTime += frameGenerator.stats.timeToGenerateQuadsMs;
        stats.totalSimplifyTime += frameGenerator.stats.timeToSimplifyQuadsMs;
        stats.totalGatherQuadsTime += frameGenerator.stats.timeToGatherQuadsMs;
        stats.totalCreateProxiesTime += frameGenerator.stats.timeToCreateProxiesMs;

        stats.totalAppendQuadsTime += frameGenerator.stats.timeToAppendQuadsMs;
        stats.totalFillQuadsIndiciesTime += frameGenerator.stats.timeToFillQuadIndicesMs;
        stats.totalCreateVertIndTime += frameGenerator.stats.timeToCreateVertIndMs;
        stats.totalCreateMeshTime += frameGenerator.stats.timeToCreateMeshMs;

        if (!generateResFrame) {
            stats.totalCompressTime += frameGenerator.stats.timeToCompress;
        }

        /*
        ============================
        Generate Residual Frame
        ============================
        */
        if (generateResFrame) {
            quadsGenerator.params.expandEdges = true;
            stats.compressedSizeBytes = frameGenerator.generateResFrame(
                meshScenes[currMeshIndex], meshScenes[prevMeshIndex],
                maskTempRT, maskFrameRT,
                remoteCamera, remoteCameraPrev,
                refFrameMeshes[currMeshIndex], maskFrameMesh,
                numProxies, numDepthOffsets
            );

            stats.totalRenderTime += frameGenerator.stats.timeToRenderMaskMs;

            stats.totalGenQuadMapTime += frameGenerator.stats.timeToGenerateQuadsMs;
            stats.totalSimplifyTime += frameGenerator.stats.timeToSimplifyQuadsMs;
            stats.totalGatherQuadsTime += frameGenerator.stats.timeToGatherQuadsMs;
            stats.totalCreateProxiesTime += frameGenerator.stats.timeToCreateProxiesMs;

            stats.totalAppendQuadsTime += frameGenerator.stats.timeToAppendQuadsMs;
            stats.totalFillQuadsIndiciesTime += frameGenerator.stats.timeToGatherQuadsMs;
            stats.totalCreateVertIndTime += frameGenerator.stats.timeToCreateVertIndMs;
            stats.totalCreateMeshTime += frameGenerator.stats.timeToCreateMeshMs;

            stats.totalCompressTime += frameGenerator.stats.timeToCompress;
        }
        stats.totalProxies += numProxies;
        stats.totalDepthOffsets += numDepthOffsets;

        maskFrameNode.visible = generateResFrame;
        currMeshIndex = (currMeshIndex + 1) % 2;
        prevMeshIndex = (prevMeshIndex + 1) % 2;

        // Only update the previous camera pose if we are not generating a Residual Frame
        if (!generateResFrame) {
            remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());
        }

        // For debugging: Generate point cloud from depth map
        if (showDepth) {
            depthMesh.update(remoteCamera, refFrameRT);
            stats.totalGenDepthTime += depthMesh.stats.genDepthTime;
        }
    }

    uint saveToFile(const Path& outputPath) {
        // Save quads
        double startTime = timeutils::getTimeMicros();
        Path filename = (outputPath / "quads").withExtension(".bin.zstd");
        std::ofstream quadsFile = std::ofstream(filename, std::ios::binary);
        quadsFile.write(quadFrame.getQuads().data(), quadFrame.getQuads().size());
        quadsFile.close();
        spdlog::info("Saved {} quads ({:.3f}MB) in {:.3f}ms",
                      stats.totalProxies, static_cast<double>(quadFrame.getQuads().size()) / BYTES_PER_MEGABYTE,
                        timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

        // Save depth offsets
        startTime = timeutils::getTimeMicros();
        Path offsetsFile = (outputPath / "depthOffsets").withExtension(".bin.zstd");
        std::ofstream depthOffsetsFile = std::ofstream(offsetsFile, std::ios::binary);
        depthOffsetsFile.write(quadFrame.getDepthOffsets().data(), quadFrame.getDepthOffsets().size());
        depthOffsetsFile.close();
        spdlog::info("Saved {} depth offsets ({:.3f}MB) in {:.3f}ms",
                     stats.totalDepthOffsets, static_cast<double>(quadFrame.getDepthOffsets().size()) / BYTES_PER_MEGABYTE,
                        timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

        // Save color buffer
        Path colorFileName = outputPath / "color";
        copyRT.saveColorAsJPG(colorFileName.withExtension(".jpg"));

        return quadFrame.getQuads().size() + quadFrame.getDepthOffsets().size();
    }

private:
    QuadFrame& quadFrame;

    FrameGenerator& frameGenerator;

    // Shaders
    ToneMapper toneMapper;
    ShowNormalsEffect showNormalsEffect;

    PerspectiveCamera remoteCameraPrev;

    // Scenes with resulting mesh
    std::vector<Scene> meshScenes;

    QuadMaterial wireframeMaterial = QuadMaterial({. baseColor = colors[0] });
    QuadMaterial maskWireframeMaterial = QuadMaterial({. baseColor = colors[colors.size()-1] });
};

} // namespace quasar


#endif // QUADS_SIMULATOR_H
