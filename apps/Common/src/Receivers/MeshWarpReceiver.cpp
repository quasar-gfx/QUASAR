#include <Receivers/MeshWarpReceiver.h>
#include <shaders_common.h>

#ifndef PLATFORM_ANDROID
#define THREADS_PER_LOCALGROUP 32
#else
#define THREADS_PER_LOCALGROUP 16
#endif

using namespace quasar;

MeshWarpReceiver::MeshWarpReceiver(
        const glm::uvec2& remoteGBufferSize,
        uint depthFactor,
        uint vertexGroupSize,
        float remoteFOV,
        const std::string& videoURL,
        const std::string& depthURL)
    : videoURL(videoURL)
    , depthURL(depthURL)
    , vertexGroupSize(vertexGroupSize)
    , depthFactor(depthFactor)
    , remoteCamera(remoteGBufferSize)
    , adjustedSize(remoteGBufferSize / vertexGroupSize)
    , videoTexture({
        .width = remoteGBufferSize.x,
        .height = remoteGBufferSize.y,
        .internalFormat = GL_RGB8,
        .format = GL_RGB,
        .type = GL_UNSIGNED_BYTE,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR,
    }, videoURL)
    , depthTexture({
        .width = remoteGBufferSize.x / depthFactor,
        .height = remoteGBufferSize.y / depthFactor,
        .internalFormat = GL_R32F,
        .format = GL_RED,
        .type = GL_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    }, depthURL)
    , meshFromBC4Shader({
        .computeCodeData = SHADER_COMMON_MESH_FROM_BC4_COMP,
        .computeCodeSize = SHADER_COMMON_MESH_FROM_BC4_COMP_len,
        .defines = {
            "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
        }
    })
    , meshMaterial({ .baseColorTexture = &videoTexture })
    , mesh({
        .maxVertices = (adjustedSize.x + 1) * (adjustedSize.y + 1),
        .maxIndices = (adjustedSize.x * adjustedSize.y + adjustedSize.x - 1) * 2 * 3,
        .material = &meshMaterial,
        .usage = GL_DYNAMIC_DRAW
    })
{
    remoteCamera.setFovyDegrees(remoteFOV);

    meshFromBC4Shader.bind();
    meshFromBC4Shader.setBool("unlinearizeDepth", true);
    meshFromBC4Shader.setVec2("depthMapSize", glm::vec2(depthTexture.width, depthTexture.height));
    meshFromBC4Shader.setUint("vertexGroupSize", vertexGroupSize);
}

void MeshWarpReceiver::loadFromFiles(const Path& dataPath) {
    // Read camera data
    Path cameraFileName = dataPath / "camera.bin";
    colorFramePose.loadFromFile(cameraFileName);
    colorFramePose.copyPoseToCamera(remoteCamera);

    // Read color data
    Path colorFileName = dataPath / "color.jpg";
    videoTexture.loadFromFile(colorFileName, true, false);

    // Read depth data
    Path depthFileName = dataPath / "depth.bc4.zstd";
    depthTexture.loadFromFile(depthFileName);

    // Update poses
    depthFramePose = colorFramePose;

    // Update mesh
    updateMesh();
}

void MeshWarpReceiver::recvData(const PoseStreamer& poseStreamer, double& elapsedTimeColor, double& elapsedTimeDepth) {
    // Render color video frame
    videoTexture.bind();
    poseIdColor = videoTexture.draw();

    // Render depth video frame
    depthTexture.bind();
    if (sync) {
        poseIdDepth = depthTexture.draw(poseIdColor);
    }
    else {
        poseIdDepth = depthTexture.draw();
    }

    // Get poses for the frames
    poseStreamer.getPose(poseIdColor, &colorFramePose, &elapsedTimeColor);
    poseStreamer.getPose(poseIdDepth, &depthFramePose, &elapsedTimeDepth);

    // Update mesh
    updateMesh();
}

void MeshWarpReceiver::updateMesh() {
    // Set shader uniforms
    meshFromBC4Shader.bind();
    {
        meshFromBC4Shader.setMat4("projection", remoteCamera.getProjectionMatrix());
        meshFromBC4Shader.setMat4("projectionInverse", remoteCamera.getProjectionMatrixInverse());
        meshFromBC4Shader.setMat4("viewColor", colorFramePose.mono.view);
        meshFromBC4Shader.setMat4("viewInverseDepth", glm::inverse(depthFramePose.mono.view));
        meshFromBC4Shader.setFloat("near", remoteCamera.getNear());
        meshFromBC4Shader.setFloat("far", remoteCamera.getFar());
    }
    {
        meshFromBC4Shader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, mesh.vertexBuffer);
        meshFromBC4Shader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, mesh.indexBuffer);
        meshFromBC4Shader.setBuffer(GL_SHADER_STORAGE_BUFFER, 2, depthTexture.bc4CompressedBuffer);
    }

    // Generate vertices and indices for mesh from depth map
    meshFromBC4Shader.dispatch(((adjustedSize.x + 1) + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                               ((adjustedSize.y + 1) + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
    meshFromBC4Shader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT |
                                    GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);
}
