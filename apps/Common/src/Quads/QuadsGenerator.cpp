#include <Quads/QuadsGenerator.h>
#include <Utils/TimeUtils.h>

#include <shaders_common.h>

#define THREADS_PER_LOCALGROUP 2 // 2x2 = 4 threads per pixel

#define MAX_PROXY_SIZE 2048

using namespace quasar;

QuadsGenerator::QuadsGenerator(QuadFrame& quadFrame)
    : quadFrame(quadFrame)
    , maxProxies(quadFrame.getSize().x * quadFrame.getSize().y)
    , sizesBuffer({
        .target = GL_SHADER_STORAGE_BUFFER,
        .dataSize = sizeof(BufferSizes),
        .numElems = 1,
        .usage = GL_DYNAMIC_COPY,
    })
    , genQuadMapShader({
        .computeCodeData = SHADER_COMMON_GEN_QUADMAP_COMP,
        .computeCodeSize = SHADER_COMMON_GEN_QUADMAP_COMP_len,
        .defines = {
            "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
        }
    })
    , simplifyQuadMapShader({
        .computeCodeData = SHADER_COMMON_SIMPLIFY_QUADMAP_COMP,
        .computeCodeSize = SHADER_COMMON_SIMPLIFY_QUADMAP_COMP_len,
        .defines = {
            "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
        }
    })
    , gatherQuadsShader({
        .computeCodeData = SHADER_COMMON_GATHER_QUADS_COMP,
        .computeCodeSize = SHADER_COMMON_GATHER_QUADS_COMP_len,
        .defines = {
            "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
        }
    })
{
    // Make sure maxProxySize is a power of 2
    glm::uvec2 maxProxySize = quadFrame.getSize();
    maxProxySize.x = 1 << static_cast<int>(glm::ceil(glm::log2(static_cast<float>(maxProxySize.x))));
    maxProxySize.y = 1 << static_cast<int>(glm::ceil(glm::log2(static_cast<float>(maxProxySize.y))));
    maxProxySize = glm::min(maxProxySize, glm::uvec2(MAX_PROXY_SIZE));

    numQuadMaps = glm::log2(static_cast<float>(glm::min(maxProxySize.x, maxProxySize.y)));

    // Create quad buffers
    quadMaps.reserve(numQuadMaps);
    quadMapSizes.reserve(numQuadMaps);

    glm::uvec2 currQuadMapSize = quadFrame.getSize();
    for (int i = 0; i < numQuadMaps; i++) {
        quadMaps.emplace_back(currQuadMapSize.x * currQuadMapSize.y);
        quadMapSizes.emplace_back(currQuadMapSize);
        currQuadMapSize = glm::max(currQuadMapSize / 2u, glm::uvec2(1u));
    }
}

QuadsGenerator::BufferSizes QuadsGenerator::getBufferSizes() {
    BufferSizes bufferSizes;
    sizesBuffer.bind();
    sizesBuffer.getData(&bufferSizes);
    return bufferSizes;
}

void QuadsGenerator::generateInitialQuadMap(
    const Texture& colorBuffer, const Texture& normalsBuffer, const Texture& depthBuffer,
    const glm::vec2& gBufferSize,
    const PerspectiveCamera& remoteCamera)
{
    /*
    ============================
    FIRST PASS: Generate quads from G-Buffer
    ============================
    */
    int closestQuadMapIdx = 0;
    for (int i = 0; i < numQuadMaps; i++) {
        if (gBufferSize.x <= quadMapSizes[i].x && gBufferSize.y <= quadMapSizes[i].y) {
            closestQuadMapIdx = i;
        }
    }

    genQuadMapShader.startTiming();

    genQuadMapShader.bind();
    {
        genQuadMapShader.setVec2("gBufferSize", gBufferSize);
        genQuadMapShader.setVec2("quadMapSize", quadMapSizes[closestQuadMapIdx]);
    }
    {
        genQuadMapShader.setTexture(normalsBuffer, 0);
        genQuadMapShader.setTexture(depthBuffer, 1);
    }
    {
        genQuadMapShader.setMat4("view", remoteCamera.getViewMatrix());
        genQuadMapShader.setMat4("projection", remoteCamera.getProjectionMatrix());
        genQuadMapShader.setMat4("viewInverse", remoteCamera.getViewMatrixInverse());
        genQuadMapShader.setMat4("projectionInverse", remoteCamera.getProjectionMatrixInverse());
        genQuadMapShader.setFloat("near", remoteCamera.getNear());
        genQuadMapShader.setFloat("far", remoteCamera.getFar());
    }
    {
        genQuadMapShader.setBool("expandEdges", params.expandEdges);
        genQuadMapShader.setBool("correctOrientation", params.correctOrientation);
        genQuadMapShader.setFloat("depthThreshold", params.depthThreshold);
        genQuadMapShader.setFloat("angleThreshold", glm::radians(params.angleThreshold));
        genQuadMapShader.setFloat("flattenThreshold", params.flattenThreshold);
    }
    {
        genQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, sizesBuffer);

        genQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, quadMaps[closestQuadMapIdx].normalSphericalsBuffer);
        genQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 2, quadMaps[closestQuadMapIdx].depthsBuffer);
        genQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 3, quadMaps[closestQuadMapIdx].metadatasBuffer);

        genQuadMapShader.setImageTexture(0, quadFrame.depthOffsets.buffer, 0, GL_FALSE, 0, GL_READ_WRITE, quadFrame.depthOffsets.buffer.internalFormat);
        genQuadMapShader.setImageTexture(1, colorBuffer, 0, GL_FALSE, 0, GL_READ_WRITE, colorBuffer.internalFormat);
    }
    genQuadMapShader.dispatch((gBufferSize.x + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                              (gBufferSize.y + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
    genQuadMapShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    genQuadMapShader.endTiming();
    stats.timeToGenerateQuadsMs = genQuadMapShader.getElapsedTime();
}

void QuadsGenerator::simplifyQuadMaps(const PerspectiveCamera& remoteCamera, const glm::vec2& gBufferSize) {
    /*
    ============================
    SECOND PASS: Simplify quad map
    ============================
    */
    int closestQuadMapIdx = 0;
    for (int i = 1; i < numQuadMaps; i++) {
        if (gBufferSize.x <= quadMapSizes[i].x && gBufferSize.y <= quadMapSizes[i].y) {
            closestQuadMapIdx = i;
        }
    }

    simplifyQuadMapShader.startTiming();

    simplifyQuadMapShader.bind();
    {
        simplifyQuadMapShader.setVec2("gBufferSize", gBufferSize);
    }
    {
        simplifyQuadMapShader.setMat4("view", remoteCamera.getViewMatrix());
        simplifyQuadMapShader.setMat4("projection", remoteCamera.getProjectionMatrix());
        simplifyQuadMapShader.setMat4("viewInverse", remoteCamera.getViewMatrixInverse());
        simplifyQuadMapShader.setMat4("projectionInverse", remoteCamera.getProjectionMatrixInverse());
        simplifyQuadMapShader.setFloat("near", remoteCamera.getNear());
        simplifyQuadMapShader.setFloat("far", remoteCamera.getFar());
    }
    {
        simplifyQuadMapShader.setBool("correctOrientation", params.correctOrientation);
        simplifyQuadMapShader.setFloat("depthThreshold", params.depthThreshold);
        simplifyQuadMapShader.setFloat("angleThreshold", glm::radians(params.angleThreshold));
        simplifyQuadMapShader.setFloat("flattenThreshold", params.flattenThreshold);
        simplifyQuadMapShader.setFloat("proxySimilarityThreshold", params.proxySimilarityThreshold);
        simplifyQuadMapShader.setInt("maxIterForceMerge", params.maxIterForceMerge);
    }
    {
        simplifyQuadMapShader.setImageTexture(0, quadFrame.depthOffsets.buffer, 0, GL_FALSE, 0, GL_READ_WRITE, quadFrame.depthOffsets.buffer.internalFormat);
    }

    int iter = 0;
    for (int i = closestQuadMapIdx + 1; i < numQuadMaps; i++) {
        auto& prevQuadMapSize = quadMapSizes[i-1];
        auto& prevQuadMaps = quadMaps[i-1];

        auto& currQuadMapSize = quadMapSizes[i];
        auto& currQuadMaps = quadMaps[i];

        {
            simplifyQuadMapShader.setInt("iter", iter);
            simplifyQuadMapShader.setVec2("inputQuadMapSize", prevQuadMapSize);
            simplifyQuadMapShader.setVec2("outputQuadMapSize", currQuadMapSize);
        }
        {
            simplifyQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, prevQuadMaps.normalSphericalsBuffer);
            simplifyQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, prevQuadMaps.depthsBuffer);
            simplifyQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 2, prevQuadMaps.metadatasBuffer);

            simplifyQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 3, currQuadMaps.normalSphericalsBuffer);
            simplifyQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 4, currQuadMaps.depthsBuffer);
            simplifyQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 5, currQuadMaps.metadatasBuffer);
        }
        simplifyQuadMapShader.dispatch((currQuadMapSize.x + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                       (currQuadMapSize.y + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
        simplifyQuadMapShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        iter++;
    }
    simplifyQuadMapShader.memoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    simplifyQuadMapShader.endTiming();
    stats.timeToSimplifyQuadsMs = simplifyQuadMapShader.getElapsedTime();
}

void QuadsGenerator::gatherOutputQuads(const glm::vec2& gBufferSize) {
    /*
    ============================
    THIRD PASS: Fill output quads buffer
    ============================
    */
    int closestQuadMapIdx = 0;
    for (int i = 1; i < numQuadMaps; i++) {
        if (gBufferSize.x <= quadMapSizes[i].x && gBufferSize.y <= quadMapSizes[i].y) {
            closestQuadMapIdx = i;
        }
    }

    gatherQuadsShader.startTiming();

    gatherQuadsShader.bind();
    {
        gatherQuadsShader.setVec2("gBufferSize", gBufferSize);
    }
    for (int i = closestQuadMapIdx; i < numQuadMaps; i++) {
        auto& currQuadMaps = quadMaps[i];
        auto& currQuadMapSize = quadMapSizes[i];

        {
            gatherQuadsShader.setVec2("quadMapSize", currQuadMapSize);
        }
        {
            gatherQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, sizesBuffer);

            gatherQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, currQuadMaps.normalSphericalsBuffer);
            gatherQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 2, currQuadMaps.depthsBuffer);
            gatherQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 3, currQuadMaps.metadatasBuffer);

            gatherQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 4, quadFrame.quadBuffers.normalSphericalsBuffer);
            gatherQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 5, quadFrame.quadBuffers.depthsBuffer);
            gatherQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 6, quadFrame.quadBuffers.metadatasBuffer);
        }
        gatherQuadsShader.dispatch((currQuadMapSize.x + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                   (currQuadMapSize.y + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
    }
    gatherQuadsShader.memoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    gatherQuadsShader.endTiming();
    stats.timeToGatherQuadsMs = gatherQuadsShader.getElapsedTime();
}

void QuadsGenerator::createProxies(
    const Texture& colorBuffer, const Texture& normalsBuffer, const Texture& depthBuffer,
    const PerspectiveCamera& remoteCamera)
{
    const glm::vec2 gBufferSize = glm::vec2(colorBuffer.width, colorBuffer.height);
    generateInitialQuadMap(colorBuffer, normalsBuffer, depthBuffer, gBufferSize, remoteCamera);
    simplifyQuadMaps(remoteCamera, gBufferSize);
    gatherOutputQuads(gBufferSize);

    QuadsGenerator::BufferSizes bufferSizes = getBufferSizes();
    quadFrame.setNumQuads(bufferSizes.numProxies);
}

void QuadsGenerator::createProxiesFromRT(
    const FrameRenderTarget& frameRT,
    const PerspectiveCamera& remoteCamera)
{
    createProxies(frameRT.colorBuffer, frameRT.normalsBuffer, frameRT.depthStencilBuffer, remoteCamera);
}

void QuadsGenerator::createProxiesFromTextures(
    const Texture& colorBuffer, const Texture& normalsBuffer, const Texture& depthBuffer,
    const PerspectiveCamera& remoteCamera)
{
    createProxies(colorBuffer, normalsBuffer, depthBuffer, remoteCamera);
}
