#include <Quads/QuadsGenerator.h>
#include <Utils/TimeUtils.h>

#include <shaders_common.h>

#define MAX_PROXY_SIZE (2 << 10)

#ifndef __ANDROID__
#define THREADS_PER_LOCALGROUP 32
#else
#define THREADS_PER_LOCALGROUP 16
#endif

using namespace quasar;

QuadsGenerator::QuadsGenerator(QuadSet& quadSet)
    : quadSet(quadSet)
    , maxProxies(quadSet.getSize().x * quadSet.getSize().y)
    , sizesBuffer({
        .target = GL_SHADER_STORAGE_BUFFER,
        .dataSize = sizeof(BufferSizes),
        .numElems = 1,
        .usage = GL_DYNAMIC_COPY,
    })
    , createQuadMapShader({
        .computeCodeData = SHADER_COMMON_CREATE_QUADMAP_COMP,
        .computeCodeSize = SHADER_COMMON_CREATE_QUADMAP_COMP_len,
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
    glm::uvec2 maxProxySize = quadSet.getSize();
    maxProxySize.x = 1 << static_cast<int>(glm::ceil(glm::log2(static_cast<float>(maxProxySize.x))));
    maxProxySize.y = 1 << static_cast<int>(glm::ceil(glm::log2(static_cast<float>(maxProxySize.y))));
    maxProxySize = glm::min(maxProxySize, glm::uvec2(MAX_PROXY_SIZE));

    numQuadMaps = glm::log2(static_cast<float>(glm::min(maxProxySize.x, maxProxySize.y)));

    // Create quad buffers
    quadMaps.reserve(numQuadMaps);
    quadMapSizes.reserve(numQuadMaps);

    glm::uvec2 currQuadMapSize = quadSet.getSize();
    for (int i = 0; i < numQuadMaps; i++) {
        quadMaps.emplace_back(currQuadMapSize.x * currQuadMapSize.y);
        quadMapSizes.emplace_back(currQuadMapSize);
        currQuadMapSize = glm::max(currQuadMapSize / 2u, glm::uvec2(1u));
    }
}

QuadsGenerator::BufferSizes QuadsGenerator::getBufferSizes() {
    BufferSizes bufferSizes;
    sizesBuffer.bind();
    void* ptr = sizesBuffer.mapToCPU(GL_MAP_READ_BIT);
    if (ptr) {
        std::memcpy(&bufferSizes, ptr, sizeof(BufferSizes));
        sizesBuffer.unmapFromCPU();
    }
    else {
        spdlog::warn("Failed to map sizesBuffer. Copying using getData");
        sizesBuffer.getData(&bufferSizes);
    }
    return bufferSizes;
}

void QuadsGenerator::generateInitialQuadMap(
    const Texture& colorTexture, const Texture& normalsTexture, const Texture& depthTexture,
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

    double startTime = timeutils::getTimeMicros();

    createQuadMapShader.bind();
    {
        createQuadMapShader.setVec2("gBufferSize", gBufferSize);
        createQuadMapShader.setVec2("quadMapSize", quadMapSizes[closestQuadMapIdx]);
    }
    {
        createQuadMapShader.setTexture(normalsTexture, 0);
        createQuadMapShader.setTexture(depthTexture, 1);
    }
    {
        createQuadMapShader.setMat4("view", remoteCamera.getViewMatrix());
        createQuadMapShader.setMat4("projection", remoteCamera.getProjectionMatrix());
        createQuadMapShader.setMat4("viewInverse", remoteCamera.getViewMatrixInverse());
        createQuadMapShader.setMat4("projectionInverse", remoteCamera.getProjectionMatrixInverse());
        createQuadMapShader.setFloat("near", remoteCamera.getNear());
        createQuadMapShader.setFloat("far", remoteCamera.getFar());
    }
    {
        createQuadMapShader.setBool("expandEdges", params.expandEdges);
        createQuadMapShader.setBool("correctOrientation", params.correctOrientation);
        createQuadMapShader.setFloat("depthThreshold", params.depthThreshold);
        createQuadMapShader.setFloat("angleThreshold", glm::radians(params.angleThreshold));
        createQuadMapShader.setFloat("flattenThreshold", params.flattenThreshold);
    }
    {
        createQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, sizesBuffer);

        createQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, quadMaps[closestQuadMapIdx].normalAndDepthBuffer);
        createQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 2, quadMaps[closestQuadMapIdx].metadatasBuffer);

        createQuadMapShader.setImageTexture(0, quadSet.depthOffsets.texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, quadSet.depthOffsets.texture.internalFormat);
        createQuadMapShader.setImageTexture(1, colorTexture, 0, GL_FALSE, 0, GL_READ_WRITE, colorTexture.internalFormat);
    }
    createQuadMapShader.dispatch((gBufferSize.x + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                 (gBufferSize.y + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
    createQuadMapShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    stats.generateQuadsTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
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

    double startTime = timeutils::getTimeMicros();

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
        simplifyQuadMapShader.setImageTexture(0, quadSet.depthOffsets.texture, 0, GL_FALSE, 0, GL_READ_WRITE, quadSet.depthOffsets.texture.internalFormat);
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
            simplifyQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, prevQuadMaps.normalAndDepthBuffer);
            simplifyQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, prevQuadMaps.metadatasBuffer);

            simplifyQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 2, currQuadMaps.normalAndDepthBuffer);
            simplifyQuadMapShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 3, currQuadMaps.metadatasBuffer);
        }
        simplifyQuadMapShader.dispatch((currQuadMapSize.x + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                       (currQuadMapSize.y + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
        simplifyQuadMapShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        iter++;
    }
    simplifyQuadMapShader.memoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    stats.simplifyQuadsTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
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

    double startTime = timeutils::getTimeMicros();

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

            gatherQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, currQuadMaps.normalAndDepthBuffer);
            gatherQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 2, currQuadMaps.metadatasBuffer);

            gatherQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 3, quadSet.quadBuffers.normalAndDepthBuffer);
            gatherQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 4, quadSet.quadBuffers.metadatasBuffer);
        }
        gatherQuadsShader.dispatch((currQuadMapSize.x + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                   (currQuadMapSize.y + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
    }
    gatherQuadsShader.memoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    stats.gatherQuadsTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
}

void QuadsGenerator::createProxies(
    const Texture& colorTexture, const Texture& normalsTexture, const Texture& depthTexture,
    const PerspectiveCamera& remoteCamera)
{
    const glm::vec2 gBufferSize = glm::vec2(colorTexture.width, colorTexture.height);
    generateInitialQuadMap(colorTexture, normalsTexture, depthTexture, gBufferSize, remoteCamera);
    simplifyQuadMaps(remoteCamera, gBufferSize);
    gatherOutputQuads(gBufferSize);

    QuadsGenerator::BufferSizes bufferSizes = getBufferSizes();
    quadSet.setNumProxies(bufferSizes.numProxies, bufferSizes.numProxiesTransparent);
}

void QuadsGenerator::createProxiesFromRT(
    const FrameRenderTarget& frameRT,
    const PerspectiveCamera& remoteCamera)
{
    createProxies(frameRT.colorTexture, frameRT.normalsTexture, frameRT.depthStencilTexture, remoteCamera);
}

void QuadsGenerator::createProxiesFromTextures(
    const Texture& colorTexture, const Texture& normalsTexture, const Texture& depthTexture,
    const PerspectiveCamera& remoteCamera)
{
    createProxies(colorTexture, normalsTexture, depthTexture, remoteCamera);
}
