#ifndef QUADS_SIMULATOR_H
#define QUADS_SIMULATOR_H

#include <DepthMesh.h>
#include <Quads/FrameGenerator.h>
#include <PostProcessing/ToneMapper.h>
#include <PostProcessing/ShowNormalsEffect.h>

namespace quasar {

class QuadsSimulator {
public:
    // Reference frame
    FrameRenderTarget refFrameRT;
    std::vector<QuadMesh> refFrameMeshes;
    std::vector<Node> refFrameNodes;
    int currMeshIndex = 0, prevMeshIndex = 1;
    ReferenceFrame referenceFrame;

    // Residual frame
    FrameRenderTarget resFrameRT;
    QuadMesh resFrameMesh;
    Node resFrameNode;
    ResidualFrame residualFrame;

    // Local objects
    std::vector<Node> refFrameNodesLocal;
    std::vector<Node> refFrameWireframesLocal;
    Node resFrameWireframeNodesLocal;

    // Depth point cloud for debugging
    DepthMesh depthMesh;
    Node depthNode;

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
        QuadSet::Sizes totalSizes;
    } stats;

    QuadsSimulator(QuadSet& quadSet, const PerspectiveCamera& remoteCamera, FrameGenerator& frameGenerator)
        : quadSet(quadSet)
        , frameGenerator(frameGenerator)
        , refFrameRT({
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
        , resFrameRT({
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
        , copyRT({
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
        // We can use less vertices and indicies for the mask since it will be sparse
        , resFrameMesh(quadSet, resFrameRT.colorTexture, MAX_NUM_PROXIES / 4)
        , depthMesh(quadSet.getSize(), glm::vec4(0.0f, 1.0f, 0.0f, 1.0f))
        , wireframeMaterial({ .baseColor = colors[0] })
        , maskWireframeMaterial({ .baseColor = colors[colors.size()-1] })
    {
        remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());
        remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());

        meshScenes.resize(2);
        refFrameMeshes.reserve(2);
        refFrameNodes.reserve(2);
        refFrameNodesLocal.reserve(2);
        refFrameWireframesLocal.reserve(2);

        for (int i = 0; i < 2; i++) {
            refFrameMeshes.emplace_back(quadSet, refFrameRT.colorTexture);

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

        resFrameNode.setEntity(&resFrameMesh);
        resFrameNode.frustumCulled = false;

        resFrameWireframeNodesLocal.setEntity(&resFrameMesh);
        resFrameWireframeNodesLocal.frustumCulled = false;
        resFrameWireframeNodesLocal.wireframe = true;
        resFrameWireframeNodesLocal.visible = false;
        resFrameWireframeNodesLocal.overrideMaterial = &maskWireframeMaterial;

        depthNode.setEntity(&depthMesh);
        depthNode.frustumCulled = false;
        depthNode.visible = false;
        depthNode.primativeType = GL_POINTS;
    }
    ~QuadsSimulator() = default;

    uint getNumTriangles() const {
        auto refMeshSizes = refFrameMeshes[currMeshIndex].getBufferSizes();
        auto maskMeshSizes = resFrameMesh.getBufferSizes();
        return (refMeshSizes.numIndices + maskMeshSizes.numIndices) / 3; // Each triangle has 3 indices
    }

    void addMeshesToScene(Scene& localScene) {
        for (int i = 0; i < 2; i++) {
            localScene.addChildNode(&refFrameNodesLocal[i]);
            localScene.addChildNode(&refFrameWireframesLocal[i]);
        }
        localScene.addChildNode(&resFrameNode);
        localScene.addChildNode(&resFrameWireframeNodesLocal);
        localScene.addChildNode(&depthNode);
    }

    void generateFrame(
        const PerspectiveCamera& remoteCamera, Scene& remoteScene,
        DeferredRenderer& remoteRenderer,
        bool generateResFrame = false, bool showNormals = false, bool showDepth = false)
    {
        auto& remoteCameraToUse = generateResFrame ? remoteCameraPrev : remoteCamera;

        // Reset stats
        stats = { 0 };

        // Render remote scene normally
        double startTime = timeutils::getTimeMicros();
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
        auto& quadsGenerator = frameGenerator.quadsGenerator;
        quadsGenerator.params.expandEdges = false;
        frameGenerator.generateRefFrame(
            refFrameRT,
            remoteCameraToUse,
            refFrameMeshes[currMeshIndex],
            referenceFrame
        );

        stats.totalGenQuadMapTime += frameGenerator.stats.timeToGenerateQuadsMs;
        stats.totalSimplifyTime += frameGenerator.stats.timeToSimplifyQuadsMs;
        stats.totalGatherQuadsTime += frameGenerator.stats.timeToGatherQuadsMs;
        stats.totalCreateProxiesTime += frameGenerator.stats.timeToCreateQuadsMs;

        stats.totalAppendQuadsTime += frameGenerator.stats.timeToAppendQuadsMs;
        stats.totalFillQuadsIndiciesTime += frameGenerator.stats.timeToFillQuadIndicesMs;
        stats.totalCreateVertIndTime += frameGenerator.stats.timeToCreateVertIndMs;
        stats.totalCreateMeshTime += frameGenerator.stats.timeToCreateMeshMs;

        if (!generateResFrame) {
            stats.totalCompressTime += frameGenerator.stats.timeToCompressMs;
        }

        /*
        ============================
        Generate Residual Frame
        ============================
        */
        if (generateResFrame) {
            quadsGenerator.params.expandEdges = true;
            frameGenerator.generateResFrame(
                meshScenes[currMeshIndex], meshScenes[prevMeshIndex],
                resFrameRT,
                remoteCamera, remoteCameraPrev,
                refFrameMeshes[currMeshIndex], resFrameMesh,
                residualFrame
            );

            stats.totalRenderTime += frameGenerator.stats.timeToRenderMaskMs;

            stats.totalGenQuadMapTime += frameGenerator.stats.timeToGenerateQuadsMs;
            stats.totalSimplifyTime += frameGenerator.stats.timeToSimplifyQuadsMs;
            stats.totalGatherQuadsTime += frameGenerator.stats.timeToGatherQuadsMs;
            stats.totalCreateProxiesTime += frameGenerator.stats.timeToCreateQuadsMs;

            stats.totalAppendQuadsTime += frameGenerator.stats.timeToAppendQuadsMs;
            stats.totalFillQuadsIndiciesTime += frameGenerator.stats.timeToGatherQuadsMs;
            stats.totalCreateVertIndTime += frameGenerator.stats.timeToCreateVertIndMs;
            stats.totalCreateMeshTime += frameGenerator.stats.timeToCreateMeshMs;

            stats.totalCompressTime += frameGenerator.stats.timeToCompressMs;
        }

        resFrameNode.visible = generateResFrame;
        currMeshIndex = (currMeshIndex + 1) % meshScenes.size();
        prevMeshIndex = (prevMeshIndex + 1) % meshScenes.size();

        // Only update the previous camera pose if we are not generating a Residual Frame
        if (!generateResFrame) {
            remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());
        }

        // For debugging: Generate point cloud from depth map
        if (showDepth) {
            depthMesh.update(remoteCamera, refFrameRT);
            stats.totalGenDepthTime += depthMesh.stats.genDepthTime;
        }

        if (!generateResFrame) {
            stats.totalSizes.numQuads += referenceFrame.getTotalNumQuads();
            stats.totalSizes.numDepthOffsets += referenceFrame.getTotalNumDepthOffsets();
            stats.totalSizes.quadsSize += referenceFrame.getTotalQuadsSize();
            stats.totalSizes.depthOffsetsSize += referenceFrame.getTotalDepthOffsetsSize();
        }
        else {
            stats.totalSizes.numQuads += residualFrame.getTotalNumQuads();
            stats.totalSizes.numDepthOffsets += residualFrame.getTotalNumDepthOffsets();
            stats.totalSizes.quadsSize += residualFrame.getTotalQuadsSize();
            stats.totalSizes.depthOffsetsSize += residualFrame.getTotalDepthOffsetsSize();
        }
    }

    uint saveToFile(const Path& outputPath) {
        // Save color
        Path colorFileName = outputPath / "color";
        copyRT.saveColorAsJPG(colorFileName.withExtension(".jpg"));

        // Save proxies
        return referenceFrame.saveToFiles(outputPath);
    }

private:
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

    QuadSet& quadSet;
    FrameGenerator& frameGenerator;

    PerspectiveCamera remoteCameraPrev;

    // Scenes with resulting meshes
    std::vector<Scene> meshScenes;

    // Holds a copy of the current frame
    FrameRenderTarget copyRT;

    // Shaders
    ToneMapper toneMapper;
    ShowNormalsEffect showNormalsEffect;

    QuadMaterial wireframeMaterial;
    QuadMaterial maskWireframeMaterial;
};

} // namespace quasar

#endif // QUADS_SIMULATOR_H
