#include <Streamers/MeshWarpStreamer.h>
#include <shaders_common.h>

#ifndef __ANDROID__
#define THREADS_PER_LOCALGROUP 32
#else
#define THREADS_PER_LOCALGROUP 16
#endif

using namespace quasar;

MeshWarpStreamer::MeshWarpStreamer(
        DeferredRenderer& remoteRenderer,
        Scene& remoteScene,
        PerspectiveCamera& remoteCamera,
        const MeshWarpStreamerCreateParams& params)
    : videoURL(params.videoURL)
    , depthURL(params.depthURL)
    , remoteRenderer(remoteRenderer)
    , remoteScene(remoteScene)
    , remoteCamera(remoteCamera)
    , adjustedSize(glm::uvec2(remoteRenderer.width, remoteRenderer.height) / params.vertexGroupSize)
    , renderTarget({
        .width = remoteRenderer.width,
        .height = remoteRenderer.height,
        .internalFormat = GL_RGBA16F,
        .format = GL_RGBA,
        .type = GL_HALF_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR,
    })
    , videoStreamerRT({
        .width = remoteRenderer.width,
        .height = remoteRenderer.height,
        .internalFormat = GL_SRGB8_ALPHA8,
        .format = GL_RGBA,
        .type = GL_UNSIGNED_BYTE,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR,
    }, videoURL, params.maxFrameRate, params.targetBitRate)
    , depthStreamerRT({
        .width = remoteRenderer.width / params.depthFactor,
        .height = remoteRenderer.height / params.depthFactor,
        .internalFormat = GL_R32F,
        .format = GL_RED,
        .type = GL_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    }, depthURL, params.maxFrameRate)
    , meshFromBC4Shader({
        .computeCodeData = SHADER_COMMON_MESH_FROM_BC4_COMP,
        .computeCodeSize = SHADER_COMMON_MESH_FROM_BC4_COMP_len,
        .defines = {
            "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
        }
    })
    , meshMaterial({ .baseColorTexture = &renderTarget.colorTexture })
    , mesh({
        .maxVertices = (adjustedSize.x + 1) * (adjustedSize.y + 1),
        .maxIndices = (adjustedSize.x * adjustedSize.y + adjustedSize.x - 1) * 2 * 3,
        .material = &meshMaterial,
        .usage = GL_DYNAMIC_DRAW
    })
    , depthEffect(remoteCamera)
{
    meshFromBC4Shader.bind();
    meshFromBC4Shader.setBool("unlinearizeDepth", true);
    meshFromBC4Shader.setVec2("depthMapSize", glm::vec2(depthStreamerRT.width, depthStreamerRT.height));
    meshFromBC4Shader.setUint("vertexGroupSize", params.vertexGroupSize);
}

RenderStats MeshWarpStreamer::generateFrame() {
    // Reset stats
    stats = {};

    // Render all objects in scene
    double startTime = timeutils::getTimeMicros();
    RenderStats renderStats = remoteRenderer.drawObjects(remoteScene, remoteCamera);

    // Copy to intermediate render target
    tonemapper.enableTonemapping(false);
    tonemapper.drawToRenderTarget(remoteRenderer, renderTarget);
    remoteRenderer.outputRT.blit(renderTarget);

    // Copy color and depth to video frames
    tonemapper.enableTonemapping(true);
    tonemapper.drawToRenderTarget(remoteRenderer, videoStreamerRT);
    depthEffect.drawToRenderTarget(remoteRenderer, depthStreamerRT);
    stats.totalRenderTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    // Compress depth map to BC4 format with ZSTD
    stats.compressedSize = depthStreamerRT.generateFrame();
    stats.totalCompressTimeMs = depthStreamerRT.stats.compressTimeMs;

    startTime = timeutils::getTimeMicros();
    meshFromBC4Shader.bind();
    {
        meshFromBC4Shader.setMat4("projection", remoteCamera.getProjectionMatrix());
        meshFromBC4Shader.setMat4("projectionInverse", remoteCamera.getProjectionMatrixInverse());
        meshFromBC4Shader.setMat4("viewColor", remoteCamera.getViewMatrix());
        meshFromBC4Shader.setMat4("viewInverseDepth", remoteCamera.getViewMatrixInverse());
        meshFromBC4Shader.setFloat("near", remoteCamera.getNear());
        meshFromBC4Shader.setFloat("far", remoteCamera.getFar());
    }
    {
        meshFromBC4Shader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, mesh.vertexBuffer);
        meshFromBC4Shader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, mesh.indexBuffer);
        meshFromBC4Shader.setBuffer(GL_SHADER_STORAGE_BUFFER, 2, depthStreamerRT.bc4CompressedBuffer);
    }
    // Dispatch compute shader to generate vertices and indices for mesh
    meshFromBC4Shader.dispatch(((adjustedSize.x + 1) + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                               ((adjustedSize.y + 1) + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
    meshFromBC4Shader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT |
                                    GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);
    stats.totalGenMeshTime = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    return renderStats;
}

void MeshWarpStreamer::sendFrame(pose_id_t poseID) {
    videoStreamerRT.sendFrame(poseID);
    depthStreamerRT.sendFrame(poseID);
}

size_t MeshWarpStreamer::writeToFiles(const Path& outputPath) {
    // Save camera data
    Pose cameraPose;
    Path cameraFileName = (outputPath / "camera").withExtension(".bin");
    cameraPose.setProjectionMatrix(remoteCamera.getProjectionMatrix());
    cameraPose.setViewMatrix(remoteCamera.getViewMatrix());
    cameraPose.writeToFile(cameraFileName);

    // Save color
    Path colorFileName = (outputPath / "color").withExtension(".jpg");
    videoStreamerRT.writeColorAsJPG(colorFileName);

    // Save depth
    Path depthFileName = (outputPath / "depth").withExtension(".bc4.zstd");
    size_t totalBytes = depthStreamerRT.writeToFile(depthFileName);

    return totalBytes;
}
